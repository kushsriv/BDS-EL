"""
Comprehensive Differential Privacy Experiment Framework
=======================================================

Experiment modules
------------------
1. DP Aggregates          — central DP with data-driven sensitivity + RDP accounting
2. RDP Composition        — compare basic, advanced, and RDP composition bounds
3. Privacy Amplification  — subsampling amplification across sampling rates
4. Membership Inference   — empirical privacy validation vs. theoretical bound
5. Local DP               — Laplace-LDP, Duchi, Piecewise vs. central DP
6. DP-ML                  — binary intrusion detection accuracy vs. epsilon
"""

import argparse
import csv
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

from dp_utils import add_dp, repeat_runs, privacy_amplification_subsampling, compute_noise_variance
from sensitivity import compute_global_sensitivity, sensitivity_report, compute_clipped_sensitivity
from rdp_accountant import RDPAccountant, rdp_laplace, rdp_gaussian, advanced_composition_bound
from membership_inference import sweep_mia, theoretical_mia_bound
from local_dp import run_ldp_experiment
from dp_ml import (prepare_kdd_data, run_dp_ml_experiment,
                   run_multiclass_experiment, compare_budget_methods,
                   encode_multiclass_labels, _CLASS_NAMES,
                   _confidence_interval, wilcoxon_signed_rank)
from dp_sgd import run_dpsgd_experiment, dp_sgd_privacy_spent
from weighted_budget import budget_comparison_report
from prv_accountant import PRVAccountant, run_prv_vs_rdp_experiment
from statistical_analysis import run_all_tables
from multi_dataset import load_dataset, detect_dataset_type
from spark_pipeline import run_spark_dp_pipeline, run_pandas_dp_pipeline


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def setup_logging(logfile):
    logging.basicConfig(
        filename=logfile,
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(console)


def parse_args():
    p = argparse.ArgumentParser(description='DP Research Framework — NSL-KDD')
    p.add_argument('--dataset', default='KDDTrain+.csv')
    p.add_argument('--features', nargs='+', default=None)
    p.add_argument('--epsilons', type=float, nargs='+',
                   default=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    p.add_argument('--mechanisms', nargs='+', default=['laplace', 'gaussian'])
    p.add_argument('--runs', type=int, default=30)
    p.add_argument('--results_dir', default='results')
    p.add_argument('--log', default='experiment.log')
    p.add_argument('--delta', type=float, default=1e-5)
    p.add_argument('--skip_ml', action='store_true',
                   help='Skip DP-ML experiment (faster runs)')
    p.add_argument('--skip_multiclass', action='store_true',
                   help='Skip 5-class IDS experiment')
    p.add_argument('--skip_budget_compare', action='store_true',
                   help='Skip budget allocation comparison')
    p.add_argument('--skip_dpsgd', action='store_true',
                   help='Skip DP-SGD experiment')
    p.add_argument('--skip_prv', action='store_true',
                   help='Skip PRV vs. RDP composition experiment')
    p.add_argument('--skip_stats', action='store_true',
                   help='Skip statistical analysis table generation')
    p.add_argument('--skip_spark_pipeline', action='store_true',
                   help='Skip the distributed Spark DP pipeline module')
    p.add_argument('--no_spark', action='store_true',
                   help='Use pandas instead of Spark (works without a Spark installation)')
    p.add_argument('--max_ml_rows', type=int, default=20000,
                   help='Max rows for DP-ML experiments (for speed)')
    p.add_argument('--ml_runs', type=int, default=30,
                   help='Repetitions for DP-ML confidence intervals')
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_csv(rows, path):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    logging.info(f'Saved {path}')


# ---------------------------------------------------------------------------
# Label detection
# ---------------------------------------------------------------------------

def detect_label_column(df_pandas):
    """
    Find the column containing NSL-KDD attack labels (strings like 'normal').
    Handles both object dtype and pandas StringDtype.
    Returns the column name or None.
    """
    for col in df_pandas.columns:
        dtype = df_pandas[col].dtype
        is_str = (dtype == object or
                  hasattr(dtype, 'name') and dtype.name in ('str', 'string', 'object'))
        if not is_str:
            continue
        try:
            sample = df_pandas[col].dropna().head(50).astype(str).str.strip().str.lower()
            if 'normal' in sample.values:
                return col
        except Exception:
            continue
    return None


def auto_detect_numerical_features(df_pandas, exclude_cols=None):
    """
    Auto-detect all usable numerical feature columns.

    Excludes: the label column, row ID columns (index, id), and
    metadata columns (difficulty_level).
    """
    if exclude_cols is None:
        exclude_cols = set()
    always_exclude = {'index', 'difficulty_level', 'id', 'label', 'class'}
    exclude_cols = set(exclude_cols) | always_exclude

    features = []
    for col in df_pandas.columns:
        if col in exclude_cols:
            continue
        numeric = pd.to_numeric(df_pandas[col], errors='coerce')
        if numeric.notna().mean() > 0.95:
            features.append(col)
    return features


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_privacy_utility(epsilons, error_dict, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    markers = ['o', 's', '^', 'D', 'v', 'P']
    for i, (label, vals) in enumerate(error_dict.items()):
        ax.plot(epsilons, vals, marker=markers[i % len(markers)], label=label, linewidth=1.8)
    ax.set_xlabel('Privacy budget ε', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _save_fig(path)


def plot_mia(epsilons, results_by_mech, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    markers = ['o', 's', '^']
    mechs = list(results_by_mech.keys())
    for i, mech in enumerate(mechs):
        rows = results_by_mech[mech]
        eps_vals = [r['epsilon'] for r in rows]
        emp_acc = [r['empirical_accuracy'] for r in rows]
        theo_acc = [r['theoretical_max_accuracy'] for r in rows]
        axes[0].plot(eps_vals, emp_acc, marker=markers[i % 3], label=f'{mech} empirical')
        axes[0].plot(eps_vals, theo_acc, '--', color=f'C{i}', label=f'{mech} DP bound')
        emp_adv = [r['empirical_advantage'] for r in rows]
        theo_adv = [r['theoretical_max_advantage'] for r in rows]
        axes[1].plot(eps_vals, emp_adv, marker=markers[i % 3], label=f'{mech} empirical')
        axes[1].plot(eps_vals, theo_adv, '--', color=f'C{i}', label=f'{mech} DP bound')

    axes[0].axhline(0.5, color='grey', linestyle=':', label='random guess')
    axes[0].set_xlabel('ε'); axes[0].set_ylabel('Attack accuracy')
    axes[0].set_title('MIA Accuracy vs. ε'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].axhline(0.0, color='grey', linestyle=':', label='no advantage')
    axes[1].set_xlabel('ε'); axes[1].set_ylabel('Attacker advantage')
    axes[1].set_title('MIA Advantage vs. ε'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    _save_fig(path)


def plot_composition(comp_rows, path):
    if not comp_rows:
        return
    n_q = [r['n_queries'] for r in comp_rows]
    basic = [r['basic_composition_epsilon'] for r in comp_rows]
    adv = [r['advanced_composition_epsilon'] for r in comp_rows]
    rdp = [r['rdp_composition_epsilon'] for r in comp_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(n_q, basic, 'o-', label='Basic composition')
    ax.semilogy(n_q, adv, 's-', label='Advanced composition')
    ax.semilogy(n_q, rdp, '^-', label='RDP composition (ours)')
    ax.set_xlabel('Number of queries k', fontsize=12)
    ax.set_ylabel('Total ε (log scale)', fontsize=12)
    ax.set_title('Privacy Budget Accumulation under Composition', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    _save_fig(path)


def plot_amplification(amp_rows, path):
    if not amp_rows:
        return
    rates = sorted(set(r['sampling_rate'] for r in amp_rows))
    mechs = sorted(set(r['mechanism'] for r in amp_rows))
    epsilons = sorted(set(r['nominal_epsilon'] for r in amp_rows))

    fig, ax = plt.subplots(figsize=(7, 4))
    for mech in mechs:
        for rate in rates:
            subset = [r for r in amp_rows if r['mechanism'] == mech and r['sampling_rate'] == rate]
            if not subset:
                continue
            eps_nom = [r['nominal_epsilon'] for r in subset]
            eps_amp = [r['amplified_epsilon'] for r in subset]
            ax.plot(eps_nom, eps_amp, marker='o',
                    label=f'{mech} q={rate:.2f}')
    ax.plot(epsilons, epsilons, 'k--', label='No amplification (q=1)')
    ax.set_xlabel('Nominal ε'); ax.set_ylabel('Effective ε after amplification')
    ax.set_title('Privacy Amplification by Subsampling', fontsize=13)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save_fig(path)


def plot_ldp_vs_central(ldp_rows, central_rows, epsilons, feature, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ldp_mechs = sorted(set(r['mechanism'] for r in ldp_rows))
    central_mechs = sorted(set(r['mechanism'] for r in central_rows))
    markers = ['o', 's', '^', 'D']

    for i, mech in enumerate(ldp_mechs):
        subset = [r for r in ldp_rows if r['mechanism'] == mech]
        eps_vals = [r['epsilon'] for r in subset]
        errs = [r['mean_error'] for r in subset]
        ax.plot(eps_vals, errs, marker=markers[i % 4], linestyle='--',
                label=f'LDP-{mech}')

    for i, mech in enumerate(central_mechs):
        subset = [r for r in central_rows if r['mechanism'] == mech]
        eps_vals = [r['epsilon'] for r in subset]
        errs = [r['mean_error'] for r in subset]
        ax.plot(eps_vals, errs, marker=markers[(i + len(ldp_mechs)) % 4],
                linestyle='-', label=f'Central-{mech}')

    ax.set_xlabel('ε'); ax.set_ylabel('Mean absolute error')
    ax.set_title(f'Local vs. Central DP: {feature}', fontsize=13)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save_fig(path)


def plot_dp_ml(ml_rows, path):
    if not ml_rows:
        return
    mechs = sorted(set(r['mechanism'] for r in ml_rows if r['mechanism'] != 'none'))
    epsilons_plot = [r['epsilon'] for r in ml_rows
                     if r['mechanism'] != 'none' and not np.isinf(r['epsilon'])]
    epsilons_plot = sorted(set(epsilons_plot))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for mech in mechs:
        subset = [r for r in ml_rows
                  if r['mechanism'] == mech and not np.isinf(r['epsilon'])]
        eps_vals = [r['epsilon'] for r in subset]
        accs = [r['accuracy'] for r in subset]
        acc_stds = [r['accuracy_std'] for r in subset]
        f1s = [r['f1_score'] for r in subset]
        f1_stds = [r['f1_std'] for r in subset]

        axes[0].errorbar(eps_vals, accs, yerr=acc_stds, marker='o',
                         capsize=4, label=f'DP-{mech}')
        axes[1].errorbar(eps_vals, f1s, yerr=f1_stds, marker='s',
                         capsize=4, label=f'DP-{mech}')

    # Baseline
    base = [r for r in ml_rows if r['mechanism'] == 'none']
    if base:
        base_acc = base[0]['accuracy']
        base_f1 = base[0]['f1_score']
        axes[0].axhline(base_acc, color='green', linestyle='--', label='No-DP baseline')
        axes[1].axhline(base_f1, color='green', linestyle='--', label='No-DP baseline')

    axes[0].set_xlabel('ε'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('DP-ML: Accuracy vs. ε', fontsize=13)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('ε'); axes[1].set_ylabel('F1 Score')
    axes[1].set_title('DP-ML: F1 vs. ε', fontsize=13)
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    _save_fig(path)


def plot_sensitivity_comparison(sens_rows, path):
    if not sens_rows:
        return
    features = [r['feature'] for r in sens_rows]
    gs = [r['global_sensitivity'] for r in sens_rows]
    ls = [r['local_sensitivity'] for r in sens_rows]

    x = np.arange(len(features))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(features) * 1.2), 4))
    ax.bar(x - w/2, gs, w, label='Global sensitivity')
    ax.bar(x + w/2, ls, w, label='Local sensitivity')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Sensitivity value')
    ax.set_title('Data-Driven Sensitivity: Global vs. Local', fontsize=13)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    _save_fig(path)


# ---------------------------------------------------------------------------
# Module 1: DP Aggregates with data-driven sensitivity
# ---------------------------------------------------------------------------

def _feature_stats_pandas(df_pandas, feature):
    """Compute count, mean, sum from pandas (Spark fallback)."""
    col = df_pandas[feature].dropna().astype(float)
    return int(len(col)), float(col.mean()), float(col.sum())


def _feature_stats_spark(df_spark, feature):
    """Compute count, mean, sum from Spark."""
    df_f = df_spark.select(F.col(feature).cast('double').alias(feature)).dropna()
    n = df_f.count()
    agg = df_f.agg(F.avg(feature).alias('avg'), F.sum(feature).alias('sum')).first()
    return n, agg['avg'], agg['sum']


def run_dp_aggregates(df_spark, df_pandas, features, epsilons, mechanisms,
                       n_runs, delta, results_dir):
    logging.info('=== Module 1: DP Aggregates (data-driven sensitivity) ===')
    results = []
    sens_rows = []

    for feature in features:
        data_col = df_pandas[feature].dropna().astype(float).values
        if len(data_col) == 0:
            continue

        # Data-driven sensitivities
        rep = sensitivity_report(data_col, 'mean')
        gs = max(rep['global_sensitivity'], 1e-10)  # guard against zero-range features
        ls = rep['local_sensitivity']
        sens_rows.append({'feature': feature, **rep, 'global_sensitivity': gs})

        # Use Spark if available, else pandas
        if df_spark is not None:
            n_rows, true_avg, sum_val = _feature_stats_spark(df_spark, feature)
        else:
            n_rows, true_avg, sum_val = _feature_stats_pandas(df_pandas, feature)

        if true_avg is None or np.isnan(true_avg):
            continue

        logging.info(f'  {feature}: GS={gs:.4g} LS={ls:.4g} mean={true_avg:.4g}')

        for mech in mechanisms:
            rdp_acc = RDPAccountant()
            for eps in epsilons:
                # POST: noise on aggregate (global sensitivity)
                def post_fn():
                    return add_dp(true_avg, gs, eps, mech, delta)
                post_vals = repeat_runs(post_fn, n_runs)
                post_errs = [abs(v - true_avg) for v in post_vals]

                # AGG: noise on sum (GS_sum = range = n * GS_mean)
                gs_sum = compute_global_sensitivity(data_col, 'sum')

                def agg_fn():
                    noisy_sum = add_dp(sum_val, gs_sum, eps, mech, delta)
                    return noisy_sum / n_rows
                agg_vals = repeat_runs(agg_fn, n_runs)
                agg_errs = [abs(v - true_avg) for v in agg_vals]

                # SHUFFLE (per-record LDP noise then average)
                def shuffle_fn():
                    noisy = [add_dp(float(x), gs, eps, mech, delta) for x in data_col[:5000]]
                    return float(np.mean(noisy))
                shuffle_vals = repeat_runs(shuffle_fn, n_runs)
                shuffle_errs = [abs(v - true_avg) for v in shuffle_vals]

                # Record RDP for this query
                if mech == 'laplace':
                    rdp_acc.add_laplace_query(eps, sensitivity=gs)
                else:
                    rdp_acc.add_gaussian_query(eps, sensitivity=gs, delta=delta)

                # Theoretical noise variance
                noise_var = compute_noise_variance(gs, eps, mech, delta)

                row = {
                    'feature': feature,
                    'mechanism': mech,
                    'epsilon': eps,
                    'global_sensitivity': gs,
                    'local_sensitivity': ls,
                    'post_mean_error': float(np.mean(post_errs)),
                    'post_std_error': float(np.std(post_errs)),
                    'agg_mean_error': float(np.mean(agg_errs)),
                    'agg_std_error': float(np.std(agg_errs)),
                    'shuffle_mean_error': float(np.mean(shuffle_errs)),
                    'shuffle_std_error': float(np.std(shuffle_errs)),
                    'noise_variance': float(noise_var) if noise_var else '',
                }
                results.append(row)

            # Plot privacy-utility for this feature/mechanism
            subset = [r for r in results if r['feature'] == feature and r['mechanism'] == mech]
            if subset:
                err_dict = {
                    f'POST (GS-calibrated)': [r['post_mean_error'] for r in subset],
                    f'AGG (GS-calibrated)': [r['agg_mean_error'] for r in subset],
                    f'SHUFFLE (GS-calibrated)': [r['shuffle_mean_error'] for r in subset],
                }
                plot_privacy_utility(
                    [r['epsilon'] for r in subset],
                    err_dict,
                    f'Privacy vs Utility: {feature} [{mech}]',
                    'Mean absolute error',
                    os.path.join(results_dir, f'{feature}_{mech}_aggregate.png'),
                )

    save_csv(results, os.path.join(results_dir, 'aggregate_results.csv'))
    save_csv(sens_rows, os.path.join(results_dir, 'sensitivity_report.csv'))
    if sens_rows:
        plot_sensitivity_comparison(
            sens_rows, os.path.join(results_dir, 'sensitivity_comparison.png')
        )
    return results


# ---------------------------------------------------------------------------
# Module 2: RDP Composition Analysis
# ---------------------------------------------------------------------------

def run_rdp_composition(mechanisms, epsilons, delta, results_dir):
    logging.info('=== Module 2: RDP Composition Analysis ===')
    rows = []
    comp_rows_by_mech = {m: [] for m in mechanisms}

    query_counts = [1, 2, 5, 10, 20, 50, 100]

    for mech in mechanisms:
        for eps in epsilons[:3]:  # use first 3 epsilon values
            acc = RDPAccountant()
            for k in query_counts:
                # Add k queries
                while len(acc.queries) < k:
                    if mech == 'laplace':
                        acc.add_laplace_query(eps, sensitivity=1.0)
                    else:
                        acc.add_gaussian_query(eps, sensitivity=1.0, delta=delta)

                comp = acc.composition_comparison(delta)
                row = {'mechanism': mech, 'nominal_epsilon': eps, **comp}
                rows.append(row)
                comp_rows_by_mech[mech].append(comp)
            acc.reset()

    save_csv(rows, os.path.join(results_dir, 'rdp_composition.csv'))

    # Plot for first mechanism and first epsilon
    for mech in mechanisms:
        subset = [r for r in rows if r['mechanism'] == mech and r['nominal_epsilon'] == epsilons[0]]
        if subset:
            plot_composition(subset, os.path.join(results_dir, f'rdp_composition_{mech}.png'))

    return rows


# ---------------------------------------------------------------------------
# Module 3: Privacy Amplification by Subsampling
# ---------------------------------------------------------------------------

def run_amplification(epsilons, mechanisms, delta, results_dir):
    logging.info('=== Module 3: Privacy Amplification by Subsampling ===')
    sampling_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
    rows = []

    for mech in mechanisms:
        for eps in epsilons:
            for q in sampling_rates:
                eps_amp, delta_amp = privacy_amplification_subsampling(eps, delta, q)
                rows.append({
                    'mechanism': mech,
                    'nominal_epsilon': eps,
                    'sampling_rate': q,
                    'amplified_epsilon': eps_amp,
                    'amplified_delta': delta_amp,
                    'amplification_ratio': eps / eps_amp if eps_amp > 0 else float('inf'),
                })

    save_csv(rows, os.path.join(results_dir, 'amplification_results.csv'))
    plot_amplification(rows, os.path.join(results_dir, 'amplification.png'))
    return rows


# ---------------------------------------------------------------------------
# Module 4: Membership Inference Attacks
# ---------------------------------------------------------------------------

def run_mia(df_pandas, features, epsilons, mechanisms, delta, results_dir):
    logging.info('=== Module 4: Membership Inference Attacks ===')
    all_rows = []

    for feature in features[:2]:  # limit to 2 features for speed
        data = df_pandas[feature].dropna().astype(float).values
        if len(data) == 0:
            continue
        sensitivity = compute_global_sensitivity(data, 'mean')
        # Use a small sample for MIA (200 records sufficient for mean query attack)
        sample = data[:200]

        mia_results = sweep_mia(
            sample, epsilons, mechanisms,
            sensitivity=sensitivity, delta=delta, n_trials=800,
        )
        for r in mia_results:
            r['feature'] = feature
        all_rows.extend(mia_results)

        # Plot
        by_mech = {}
        for mech in mechanisms:
            by_mech[mech] = [r for r in mia_results if r['mechanism'] == mech]
        plot_mia(epsilons, by_mech,
                 os.path.join(results_dir, f'mia_{feature}.png'))

    save_csv(all_rows, os.path.join(results_dir, 'mia_results.csv'))
    return all_rows


# ---------------------------------------------------------------------------
# Module 5: Local DP vs. Central DP
# ---------------------------------------------------------------------------

def run_local_dp(df_pandas, features, epsilons, mechanisms, delta, results_dir):
    logging.info('=== Module 5: Local DP vs. Central DP ===')
    ldp_mechs = ['laplace', 'duchi', 'piecewise']
    all_ldp = []
    all_central = []

    for feature in features[:2]:
        data = df_pandas[feature].dropna().astype(float).values
        if len(data) == 0:
            continue
        sensitivity = compute_global_sensitivity(data, 'mean')
        true_mean = float(np.mean(data))
        # Subsample for speed (LDP applies per-record)
        sample = data[:3000]

        ldp_rows = []
        for ldp_mech in ldp_mechs:
            for eps in epsilons:
                r = run_ldp_experiment(sample, eps, mechanism=ldp_mech,
                                       sensitivity=sensitivity, n_runs=20)
                if r:
                    r['feature'] = feature
                    ldp_rows.append(r)
                    all_ldp.append(r)

        central_rows = []
        for mech in mechanisms:
            for eps in epsilons:
                def central_fn():
                    return add_dp(true_mean, sensitivity, eps, mech, delta)
                errs = [abs(central_fn() - true_mean) for _ in range(20)]
                r = {
                    'feature': feature,
                    'mechanism': mech,
                    'epsilon': eps,
                    'privacy_model': 'central',
                    'mean_error': float(np.mean(errs)),
                    'std_error': float(np.std(errs)),
                }
                central_rows.append(r)
                all_central.append(r)

        plot_ldp_vs_central(
            ldp_rows, central_rows, epsilons, feature,
            os.path.join(results_dir, f'ldp_vs_central_{feature}.png'),
        )

    save_csv(all_ldp, os.path.join(results_dir, 'ldp_results.csv'))
    save_csv(all_central, os.path.join(results_dir, 'central_dp_results.csv'))
    return all_ldp, all_central


# ---------------------------------------------------------------------------
# Module 6: DP-ML — Intrusion Detection
# ---------------------------------------------------------------------------

def run_dp_ml(df_pandas, features, label_col, epsilons, mechanisms, delta,
              results_dir, max_rows=20000, n_runs=30):
    logging.info('=== Module 6: DP-ML — Binary Intrusion Detection ===')
    if label_col is None:
        logging.warning('No label column found; skipping DP-ML experiment.')
        return []

    logging.info(f'  Using label column: "{label_col}"')
    X_train, X_test, y_train, y_test = prepare_kdd_data(
        df_pandas, features, label_col,
        test_size=0.3, max_rows=max_rows,
    )
    logging.info(f'  Train: {X_train.shape}  Test: {X_test.shape}  '
                 f'Attack rate train={y_train.mean():.2%} test={y_test.mean():.2%}')

    all_ml = []
    for mech in mechanisms:
        # Standard uniform budget (clipped sensitivity)
        ml_rows, base_acc = run_dp_ml_experiment(
            X_train, X_test, y_train, y_test,
            epsilons, mechanism=mech, n_runs=n_runs, delta=delta,
            sensitivity_method='clipped', budget_method='uniform',
        )
        all_ml.extend(ml_rows)
        # MI-weighted budget (novel)
        ml_rows_mi, _ = run_dp_ml_experiment(
            X_train, X_test, y_train, y_test,
            epsilons, mechanism=mech, n_runs=n_runs, delta=delta,
            sensitivity_method='clipped', budget_method='mi',
        )
        all_ml.extend(ml_rows_mi)
        logging.info(f'  {mech}: baseline={base_acc:.4f}  '
                     f'ε={epsilons[-1]} uniform={ml_rows[-1]["accuracy"]:.4f}'
                     f' mi_weighted={ml_rows_mi[-1]["accuracy"]:.4f}')

    save_csv(all_ml, os.path.join(results_dir, 'dp_ml_results.csv'))
    plot_dp_ml(all_ml, os.path.join(results_dir, 'dp_ml.png'))
    return all_ml


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_budget_comparison(df_pandas, features, label_col, epsilons, delta,
                           max_rows, ml_runs, results_dir):
    """Module 7: Compare uniform vs. MI/variance/SNR-weighted budget allocation."""
    logging.info('=== Module 7: Budget Allocation Comparison ===')
    if label_col is None:
        logging.warning('No label column; skipping budget comparison.')
        return []

    X_train, X_test, y_train, y_test = prepare_kdd_data(
        df_pandas, features, label_col, max_rows=max_rows
    )
    logging.info(f'  Budget comparison: Train={X_train.shape} Test={X_test.shape}')

    all_rows = compare_budget_methods(
        X_train, X_test, y_train, y_test,
        epsilons=epsilons[:4],   # first 4 epsilons for speed
        mechanism='laplace',
        n_runs=ml_runs // 3,    # fewer runs for comparison
        delta=delta,
    )
    save_csv(all_rows, os.path.join(results_dir, 'budget_comparison.csv'))

    # Plot: accuracy vs epsilon, one line per budget method
    dp_rows = [r for r in all_rows if r['mechanism'] != 'none']
    methods = sorted(set(r['budget_method'] for r in dp_rows))
    eps_vals = sorted(set(r['epsilon'] for r in dp_rows))
    err_dict = {}
    for m in methods:
        err_dict[m] = [
            np.mean([r['accuracy'] for r in dp_rows
                     if r['budget_method'] == m and r['epsilon'] == e])
            for e in eps_vals
        ]
    plot_privacy_utility(
        eps_vals, err_dict,
        'Budget Allocation: Accuracy vs. ε (Laplace)',
        'Accuracy', os.path.join(results_dir, 'budget_comparison.png'),
    )
    return all_rows


def run_multiclass(df_pandas, features, label_col, epsilons, delta,
                   max_rows, ml_runs, results_dir):
    """Module 8: 5-class IDS (Normal / DoS / Probe / R2L / U2R)."""
    logging.info('=== Module 8: Multi-Class IDS (5 categories) ===')
    if label_col is None:
        logging.warning('No label column; skipping multi-class experiment.')
        return []

    X_train, X_test, y_train_mc, y_test_mc = prepare_kdd_data(
        df_pandas, features, label_col, max_rows=max_rows, multiclass=True
    )
    logging.info(f'  Multi-class: Train={X_train.shape} classes={np.unique(y_train_mc)}')
    class_dist = {_CLASS_NAMES[k]: int(np.sum(y_train_mc == k))
                  for k in range(len(_CLASS_NAMES))}
    logging.info(f'  Class distribution (train): {class_dist}')

    all_rows = []
    for mech in ['laplace']:
        for budget in ['uniform', 'mi']:
            rows, base_acc, _ = run_multiclass_experiment(
                X_train, X_test, y_train_mc, y_test_mc,
                epsilons=epsilons[:4],
                mechanism=mech,
                n_runs=max(5, ml_runs // 6),
                delta=delta,
                sensitivity_method='clipped',
                budget_method=budget,
            )
            for r in rows:
                r['mechanism'] = mech
            all_rows.extend(rows)
            logging.info(f'  {mech}/{budget}: baseline={base_acc:.4f}')

    save_csv(all_rows, os.path.join(results_dir, 'multiclass_results.csv'))
    return all_rows


def run_dp_sgd_module(df_pandas, features, label_col, epsilons, delta,
                       max_rows, ml_runs, results_dir):
    """Module 9: DP-SGD vs. input perturbation."""
    logging.info('=== Module 9: DP-SGD (Abadi et al. 2016) ===')
    if label_col is None:
        logging.warning('No label column; skipping DP-SGD experiment.')
        return []

    X_train, X_test, y_train, y_test = prepare_kdd_data(
        df_pandas, features, label_col, max_rows=max_rows
    )
    logging.info(f'  DP-SGD: Train={X_train.shape} Test={X_test.shape}')

    # Log theoretical accounting for reference
    for eps in epsilons[:3]:
        sigma_est = 1.5  # approximate; actual is found by binary search
        n = X_train.shape[0]
        actual_eps, alpha = dp_sgd_privacy_spent(n, 256, 30, sigma_est, delta)
        logging.info(f'  DP-SGD σ={sigma_est:.2f}: ε_actual≈{actual_eps:.3f} (α={alpha})')

    sgd_rows, base_acc = run_dpsgd_experiment(
        X_train, X_test, y_train, y_test,
        epsilons=epsilons[:4],
        delta=delta,
        n_runs=max(3, ml_runs // 10),
        batch_size=256,
        n_epochs=30,
        clip_norm=1.0,
    )
    logging.info(f'  DP-SGD baseline={base_acc:.4f}  '
                 f'ε={epsilons[min(3, len(epsilons)-1)]} acc={sgd_rows[-1]["accuracy"]:.4f}')

    save_csv(sgd_rows, os.path.join(results_dir, 'dpsgd_results.csv'))
    return sgd_rows


def run_clipped_sensitivity_analysis(df_pandas, features, results_dir):
    """Module 10: Clipped sensitivity comparison across all features."""
    logging.info('=== Module 10: Clipped Sensitivity Analysis ===')
    rows = []
    for feature in features:
        col = pd.to_numeric(df_pandas[feature], errors='coerce').dropna().values
        if len(col) == 0:
            continue
        for p in [90.0, 95.0, 99.0, 99.5]:
            cs = compute_clipped_sensitivity(col, 'mean', clip_percentile=p)
            rows.append({
                'feature': feature,
                'clip_percentile': p,
                'gs_raw': cs['gs_raw'],
                'gs_clipped': cs['gs_clipped'],
                'clip_threshold': cs['clip_threshold'],
                'bias_bound': cs['bias_bound'],
                'relative_bias_pct': cs['relative_bias_pct'],
                'noise_reduction': cs['noise_reduction'],
            })
        if rows:
            r99 = next(r for r in rows if r['feature'] == feature
                       and r['clip_percentile'] == 99.0)
            logging.info(f'  {feature}: noise_reduction@p99={r99["noise_reduction"]:.1f}×')

    save_csv(rows, os.path.join(results_dir, 'clipped_sensitivity.csv'))

    # Bar plot: noise reduction at p=99 across features
    features_plot = [r['feature'] for r in rows if r['clip_percentile'] == 99.0]
    nr = [r['noise_reduction'] for r in rows if r['clip_percentile'] == 99.0]
    if features_plot:
        fig, ax = plt.subplots(figsize=(max(10, len(features_plot) * 0.6), 4))
        ax.bar(range(len(features_plot)), nr, color='#3b82f6')
        ax.set_xticks(range(len(features_plot)))
        ax.set_xticklabels(features_plot, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Noise reduction factor (×)')
        ax.set_title('Clipped Sensitivity: Noise Reduction at 99th Percentile')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)
        _save_fig(os.path.join(results_dir, 'clipped_sensitivity.png'))
    return rows


def run_prv_module(epsilons, mechanisms, delta, results_dir):
    """Module 11: PRV accountant vs. RDP — tighter composition bounds."""
    logging.info('=== Module 11: PRV vs. RDP Composition (Gopi et al. 2021) ===')
    eps_subset = epsilons[:2]  # first 2 epsilons (PRV is slow for many)
    k_values = [1, 2, 5, 10, 20, 50, 100]
    rows = []

    for mech in mechanisms[:1]:  # laplace only for PRV (Gaussian is slower)
        mech_rows = run_prv_vs_rdp_experiment(
            epsilons_per_query=eps_subset,
            k_values=k_values,
            delta=delta,
            mechanism=mech,
            grid_size=2048,
            epsilon_max=max(epsilons) * len(k_values) + 5.0,
        )
        for r in mech_rows:
            r['mechanism'] = mech
            prv = r['prv_epsilon']
            rdp = r['rdp_epsilon']
            logging.info(
                f'  k={r["n_queries"]:3d}  ε₀={r["per_query_epsilon"]}  '
                f'RDP={rdp:.3f}  PRV={prv:.3f}  '
                f'tighter={r["prv_tightening_pct"]:.1f}%'
            )
        rows.extend(mech_rows)

    save_csv(rows, os.path.join(results_dir, 'prv_composition.csv'))

    # Plot
    if rows:
        k_vals = sorted(set(r['n_queries'] for r in rows))
        eps0 = rows[0]['per_query_epsilon']
        subset = [r for r in rows if r['per_query_epsilon'] == eps0]
        prv_vals = [r['prv_epsilon'] for r in subset if r['n_queries'] in k_vals]
        rdp_vals = [r['rdp_epsilon'] for r in subset if r['n_queries'] in k_vals]
        basic_vals = [r['basic_epsilon'] for r in subset if r['n_queries'] in k_vals]
        k_plot = [r['n_queries'] for r in subset if r['n_queries'] in k_vals]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(k_plot, basic_vals, 'o-', label='Basic composition')
        ax.semilogy(k_plot, rdp_vals, 's-', label='RDP (Mironov 2017)')
        ax.semilogy(k_plot, prv_vals, '^-', color='green', label='PRV (Gopi et al. 2021)')
        ax.set_xlabel('Number of queries k', fontsize=12)
        ax.set_ylabel('Total ε (log scale)', fontsize=12)
        ax.set_title(f'Composition Bounds: PRV vs. RDP (ε₀={eps0}, δ={delta})', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        _save_fig(os.path.join(results_dir, 'prv_vs_rdp.png'))

    return rows


def _run_spark_dp_pipeline_module(spark, df_pandas, features, label_col,
                                   data_path, eps_list, delta, results_dir,
                                   use_spark):
    """
    Module: Distributed Spark DP Pipeline
    ======================================
    Runs the full importance-weighted DP noise injection pipeline:
      - Distributed MI computation (on Spark workers) or single-machine fallback
      - Per-feature ε allocation via weighted budget
      - Laplace noise injection distributed across partitions
    Saves per-feature ε allocation and noise metadata to CSV.
    """
    logging.info('=== Module: Spark Distributed DP Pipeline ===')
    rows = []

    for method in ['mi', 'variance', 'snr', 'uniform']:
        for eps in eps_list:
            try:
                if use_spark and spark is not None:
                    logging.info(f'  [Spark] method={method} ε={eps} — distributed run')
                    result = run_spark_dp_pipeline(
                        spark, data_path, label_col,
                        total_eps=eps, method=method,
                        mechanism='laplace', clip_pct=99.0, delta=delta,
                        feature_cols=features,
                    )
                else:
                    logging.info(f'  [pandas] method={method} ε={eps} — single-machine fallback')
                    result = run_pandas_dp_pipeline(
                        df_pandas, label_col, features,
                        total_eps=eps, method=method,
                        mechanism='laplace', clip_pct=99.0, delta=delta,
                    )

                # Top-3 features by allocated ε
                top3 = sorted(result['epsilons'].items(), key=lambda x: -x[1])[:3]
                top3_str = ' | '.join(f'{f}={e:.4f}' for f, e in top3)

                rows.append({
                    'method':        method,
                    'epsilon_total': eps,
                    'mode':          'spark' if (use_spark and spark is not None) else 'pandas',
                    'n_features':    len(features),
                    'top3_eps':      top3_str,
                    'min_eps':       min(result['epsilons'].values()),
                    'max_eps':       max(result['epsilons'].values()),
                })

            except Exception as ex:
                logging.warning(f'  Spark pipeline failed method={method} ε={eps}: {ex}')

    if rows:
        save_csv(rows, os.path.join(results_dir, 'spark_dp_pipeline.csv'))
        logging.info(f'Saved results/spark_dp_pipeline.csv  ({len(rows)} rows)')


def main():
    args = parse_args()
    setup_logging(args.log)
    ensure_dir(args.results_dir)

    use_spark = SPARK_AVAILABLE and not args.no_spark

    # Auto-detect dataset type and use appropriate loader
    dataset_type = detect_dataset_type(args.dataset)
    logging.info(f'Detected dataset type: {dataset_type}')

    if dataset_type != 'NSL-KDD' and os.path.exists(args.dataset):
        # Use multi_dataset loader for UNSW-NB15, CIC-IDS-2017, etc.
        logging.info(f'Using multi_dataset loader for {dataset_type}')
        df_pandas, label_col, dataset_info = load_dataset(args.dataset)
        logging.info(f'Dataset: {dataset_info["dataset"]} — '
                     f'{dataset_info["n_rows"]:,} rows, {dataset_info["n_cols"]} cols')
        spark = None
        df_spark = None
    elif use_spark:
        spark = SparkSession.builder \
            .appName('DP-Research-Framework') \
            .config('spark.driver.memory', '4g') \
            .getOrCreate()
        spark.sparkContext.setLogLevel('ERROR')
        logging.info(f'Loading dataset via Spark: {args.dataset}')
        df_spark = spark.read.csv(args.dataset, header=True, inferSchema=True)
        df_pandas = df_spark.toPandas()
        label_col = detect_label_column(df_pandas)
    else:
        spark = None
        df_spark = None
        if not SPARK_AVAILABLE:
            logging.info('PySpark not installed — running in pandas-only mode.')
        else:
            logging.info('--no_spark flag set — running in pandas-only mode.')
        logging.info(f'Loading dataset via pandas: {args.dataset}')
        df_pandas = pd.read_csv(args.dataset)
        label_col = detect_label_column(df_pandas)

    logging.info(f'Dataset shape: {df_pandas.shape}')
    logging.info(f'Using label column: {label_col}')

    # Feature selection: auto-detect all numerical columns if not specified
    if args.features:
        features = [f for f in args.features if f in df_pandas.columns]
        missing = [f for f in args.features if f not in df_pandas.columns]
        if missing:
            logging.warning(f'Skipped missing features: {missing}')
    else:
        exclude = {label_col} if label_col else set()
        features = auto_detect_numerical_features(df_pandas, exclude_cols=exclude)
        logging.info(f'Auto-detected {len(features)} numerical features')

    # Coerce all feature columns to float
    for f in features:
        df_pandas[f] = pd.to_numeric(df_pandas[f], errors='coerce')

    logging.info(f'Using {len(features)} features: {features}')

    eps = args.epsilons
    mechs = args.mechanisms
    delta = args.delta

    # ── Run all modules ──────────────────────────────────────────────────────

    run_dp_aggregates(df_spark, df_pandas, features[:8], eps, mechs,
                      args.runs, delta, args.results_dir)

    run_rdp_composition(mechs, eps, delta, args.results_dir)

    run_amplification(eps, mechs, delta, args.results_dir)

    run_mia(df_pandas, features, eps, mechs, delta, args.results_dir)

    run_local_dp(df_pandas, features[:4], eps, mechs, delta, args.results_dir)

    run_clipped_sensitivity_analysis(df_pandas, features, args.results_dir)

    if not args.skip_ml:
        run_dp_ml(df_pandas, features, label_col, eps, mechs,
                  delta, args.results_dir,
                  max_rows=args.max_ml_rows, n_runs=args.ml_runs)

    if not args.skip_budget_compare:
        run_budget_comparison(
            df_pandas, features, label_col, eps, delta,
            args.max_ml_rows, args.ml_runs, args.results_dir
        )

    if not args.skip_multiclass:
        run_multiclass(
            df_pandas, features, label_col, eps, delta,
            args.max_ml_rows, args.ml_runs, args.results_dir
        )

    if not args.skip_dpsgd:
        run_dp_sgd_module(
            df_pandas, features, label_col, eps, delta,
            args.max_ml_rows, args.ml_runs, args.results_dir
        )

    if not args.skip_prv:
        run_prv_module(eps, mechs, delta, args.results_dir)

    if not args.skip_stats:
        logging.info('=== Statistical Analysis Tables ===')
        run_all_tables(args.results_dir)

    # ── Spark distributed DP pipeline (if Spark available) ──────────────────
    if not args.skip_spark_pipeline:
        logging.info('=== Spark Distributed DP Pipeline ===')
        _run_spark_dp_pipeline_module(
            spark, df_pandas, features, label_col, args.dataset,
            eps, delta, args.results_dir, use_spark
        )

    if spark is not None:
        spark.stop()
    logging.info('All experiments complete.  Results in: ' + args.results_dir)


if __name__ == '__main__':
    main()
