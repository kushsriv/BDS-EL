"""
export_results.py — Convert results/*.csv to frontend/public/results_data.json

Run automatically at the end of main.py, or manually:
    python export_results.py

The frontend fetches /results_data.json on load and uses live data.
Falls back to hardcoded dpmath.js data if the file is absent.
"""

import os
import json
import datetime
import pandas as pd


def _safe_float(val):
    try:
        v = float(val)
        return None if (v != v) else v  # NaN check
    except (TypeError, ValueError):
        return None


def export_results(results_dir='results', output_path='frontend/public/results_data.json'):
    data = {}

    # ── 1. Sensitivity (Section 2) ─────────────────────────────────────────
    path = os.path.join(results_dir, 'sensitivity_report.csv')
    if os.path.exists(path):
        df = pd.read_csv(path).sort_values('global_sensitivity', ascending=False)
        data['sensitivity'] = [
            {
                'feature':           r.feature,
                'globalSensitivity': _safe_float(r.global_sensitivity),
                'localSensitivity':  _safe_float(r.local_sensitivity),
            }
            for r in df.itertuples()
        ]

    # ── 2. Amplification (Section 4) ───────────────────────────────────────
    path = os.path.join(results_dir, 'amplification_results.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        # pivot: one row per nominal_epsilon, columns = sampling rates
        rows = {}
        for r in df.itertuples():
            key = _safe_float(r.nominal_epsilon)
            if key not in rows:
                rows[key] = {'epsilon': key}
            rows[key][f'q={r.sampling_rate}'] = _safe_float(r.amplified_epsilon)
        data['amplification'] = list(rows.values())

    # ── 3. MIA (Section 5) ─────────────────────────────────────────────────
    path = os.path.join(results_dir, 'mia_results.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        lap = df[df['mechanism'] == 'laplace']
        # one entry per (epsilon, feature) — collapse by averaging across features
        agg = lap.groupby('epsilon').agg(
            empiricalAccuracy=('empirical_accuracy', 'mean'),
            theoreticalBound=('theoretical_max_accuracy', 'mean'),
        ).reset_index().sort_values('epsilon')
        data['mia'] = [
            {
                'epsilon':          _safe_float(r.epsilon),
                'empiricalAccuracy': _safe_float(r.empiricalAccuracy),
                'theoreticalBound':  _safe_float(r.theoreticalBound),
            }
            for r in agg.itertuples()
        ]

    # ── 4. DP-ML budget collapse (Section 7) ───────────────────────────────
    path = os.path.join(results_dir, 'dp_ml_results.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        no_dp = df[df['privacy_type'] == 'no_dp']
        baseline_acc = _safe_float(no_dp['accuracy'].values[0]) if len(no_dp) else 0.9477

        dp = df[df['privacy_type'] == 'central_dp']
        lap_uni = dp[(dp['mechanism'] == 'laplace') & (dp['budget_method'] == 'uniform')].sort_values('epsilon')
        gau_uni = dp[(dp['mechanism'] == 'gaussian') & (dp['budget_method'] == 'uniform')].sort_values('epsilon')

        eps_vals = sorted(lap_uni['epsilon'].unique())
        ml_chart = []
        for eps in eps_vals:
            lr = lap_uni[lap_uni['epsilon'] == eps]
            gr = gau_uni[gau_uni['epsilon'] == eps]
            ml_chart.append({
                'epsilon':     eps,
                'laplace_acc':  _safe_float(lr['accuracy'].values[0]) if len(lr) else None,
                'gaussian_acc': _safe_float(gr['accuracy'].values[0]) if len(gr) else None,
            })
        data['ml'] = ml_chart
        data['ml_baseline'] = baseline_acc

        # accuracy by budget method at max ε (for Section 13)
        eps_max = dp['epsilon'].max()
        mi_row  = dp[(dp['mechanism'] == 'laplace') & (dp['budget_method'] == 'mi')   & (dp['epsilon'] == eps_max)]
        uni_row = dp[(dp['mechanism'] == 'laplace') & (dp['budget_method'] == 'uniform') & (dp['epsilon'] == eps_max)]
        data['budget_accuracy'] = {
            'no_dp':            baseline_acc,
            'uniform_max_eps':  _safe_float(uni_row['accuracy'].values[0]) if len(uni_row) else None,
            'mi_max_eps':       _safe_float(mi_row['accuracy'].values[0])  if len(mi_row)  else None,
            'max_eps':          _safe_float(eps_max),
        }

    # ── 5. DP-SGD (Section 8) ──────────────────────────────────────────────
    path = os.path.join(results_dir, 'dpsgd_results.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        dp = df[df['privacy_type'] == 'central_dp'].sort_values('epsilon')
        data['dpsgd'] = [
            {
                'epsilon':  _safe_float(r.epsilon),          # nominal (for x-axis label)
                'accuracy': _safe_float(r.accuracy),
                'std':      _safe_float(r.accuracy_std),
            }
            for r in dp.itertuples()
        ]

    # ── 6. Clipped Sensitivity (Section 9) ─────────────────────────────────
    path = os.path.join(results_dir, 'clipped_sensitivity.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        p99 = df[df['clip_percentile'] == 99.0].sort_values('noise_reduction', ascending=False)
        # deduplicate by feature (keep highest noise_reduction row)
        seen = set()
        rows = []
        for r in p99.itertuples():
            if r.feature not in seen:
                seen.add(r.feature)
                rows.append({
                    'feature':         r.feature,
                    'noise_reduction': _safe_float(r.noise_reduction),
                    'gs_raw':          _safe_float(r.gs_raw),
                    'gs_clipped':      _safe_float(r.gs_clipped),
                    'bias_bound':      _safe_float(r.bias_bound),
                })
        data['clipped_sens'] = rows

    # ── 7. PRV Accountant (Section 10) ─────────────────────────────────────
    path = os.path.join(results_dir, 'prv_composition.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        lap = df[df['mechanism'] == 'laplace']
        # prefer per_query_epsilon=0.5; fallback to first available
        eps05 = lap[lap['per_query_epsilon'] == 0.5]
        subset = eps05 if len(eps05) else lap[lap['per_query_epsilon'] == lap['per_query_epsilon'].min()]
        subset = subset.sort_values('n_queries')
        data['prv'] = [
            {
                'k':        int(r.n_queries),
                'prv':      _safe_float(r.prv_epsilon),
                'rdp':      _safe_float(r.rdp_epsilon),
                'advanced': _safe_float(r.advanced_epsilon),
                'basic':    _safe_float(r.basic_epsilon),
            }
            for r in subset.itertuples()
        ]

    # ── 8. Multi-class IDS (Section 11) ────────────────────────────────────
    path = os.path.join(results_dir, 'multiclass_results.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        classes = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        no_dp = df[df['privacy_type'] == 'no_dp']
        dp_row = df[df['privacy_type'] == 'central_dp']
        if len(no_dp) and len(dp_row):
            data['multiclass'] = {
                'baseline':  {c: _safe_float(no_dp.iloc[0][f'acc_{c}'])  for c in classes},
                'collapsed': {c: _safe_float(dp_row.iloc[0][f'acc_{c}']) for c in classes},
            }

    # ── 9. Spark Pipeline (Section 13) ─────────────────────────────────────
    path = os.path.join(results_dir, 'spark_dp_pipeline.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        METHOD_LABELS = {
            'mi':       'MI-Weighted',
            'snr':      'SNR-Heuristic',
            'variance': 'Variance-Weighted',
            'uniform':  'Uniform',
        }
        METHOD_COLORS = {
            'mi':       '#3b82f6',
            'snr':      '#8b5cf6',
            'variance': '#f59e0b',
            'uniform':  '#475569',
        }

        # methodComparison at ε_total = 1.0
        mc = []
        for m in ['mi', 'snr', 'variance', 'uniform']:
            row = df[(df['method'] == m) & (df['epsilon_total'] == 1.0)]
            if len(row):
                r = row.iloc[0]
                mn = _safe_float(r['min_eps'])
                mx = _safe_float(r['max_eps'])
                ratio = round(mx / mn) if mn and mn > 0 else 1
                mc.append({'method': METHOD_LABELS[m], 'max': mx, 'min': mn,
                           'ratio': ratio, 'color': METHOD_COLORS[m]})
        data['spark_method_comparison'] = mc

        # budgetByEps: mi_max vs uniform max across epsilon_total values
        budget_by_eps = []
        for eps in sorted(df['epsilon_total'].unique()):
            mi_r  = df[(df['method'] == 'mi')      & (df['epsilon_total'] == eps)]
            uni_r = df[(df['method'] == 'uniform')  & (df['epsilon_total'] == eps)]
            if len(mi_r) and len(uni_r):
                budget_by_eps.append({
                    'eps':     _safe_float(eps),
                    'mi_max':  _safe_float(mi_r.iloc[0]['max_eps']),
                    'uniform': _safe_float(uni_r.iloc[0]['max_eps']),
                })
        data['spark_budget_by_eps'] = budget_by_eps

    # ── 9b. Per-feature budget (Section 13 bar chart) ──────────────────────
    path = os.path.join(results_dir, 'feature_budget.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        mi_at_1  = df[(df['method'] == 'mi')      & (df['epsilon_total'] == 1.0)].sort_values('epsilon_allocated', ascending=False)
        uni_at_1 = df[(df['method'] == 'uniform')  & (df['epsilon_total'] == 1.0)]
        uni_val  = _safe_float(uni_at_1['epsilon_allocated'].mean()) if len(uni_at_1) else None
        data['spark_feature_budget'] = [
            {'feature': r.feature, 'mi': _safe_float(r.epsilon_allocated), 'uniform': uni_val}
            for r in mi_at_1.itertuples()
        ]

    # ── 10. RDP Composition (Section 3) ────────────────────────────────────
    path = os.path.join(results_dir, 'rdp_composition.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        lap = df[df['mechanism'] == 'laplace']
        # group by nominal_epsilon → used to seed the interactive slider
        eps_vals = sorted(lap['nominal_epsilon'].unique())
        rdp_data = {}
        for eps in eps_vals:
            rows = lap[lap['nominal_epsilon'] == eps].sort_values('n_queries')
            rdp_data[str(eps)] = [
                {
                    'queries':   int(r.n_queries),
                    'basic':     _safe_float(r.basic_composition_epsilon),
                    'advanced':  _safe_float(r.advanced_composition_epsilon),
                    'rdp':       _safe_float(r.rdp_composition_epsilon),
                }
                for r in rows.itertuples()
            ]
        data['rdp'] = rdp_data

    # ── Metadata ───────────────────────────────────────────────────────────
    data['generated_at'] = datetime.datetime.now().isoformat()
    data['status'] = 'live'

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    present = [k for k in data if k not in ('generated_at', 'status')]
    print(f'INFO: Exported live results → {output_path}  ({len(present)} sections)')
    return data


if __name__ == '__main__':
    export_results()
