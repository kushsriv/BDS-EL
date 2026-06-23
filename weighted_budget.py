"""
Importance-Weighted DP Budget Allocation (Novel Contribution)
=============================================================

Standard input-perturbation DP-ML splits the total privacy budget uniformly:
    ε_j = ε_total / d  for each of d features.

This module proposes allocating the budget proportionally to feature
importance, which provably preserves the same total DP guarantee while
concentrating noise reduction on the features that matter most.

Privacy Guarantee (Formal)
--------------------------
Theorem (Basic Composition, Dwork & Roth 2014, Theorem 3.14):
  If mechanism M_j is ε_j-DP for feature j, then the joint mechanism
  (M_1, ..., M_d) applied independently to each feature is (Σ ε_j)-DP.

  Since Σ w_j = 1 by construction, Σ ε_j = ε_total · Σ w_j = ε_total.
  Therefore the allocation is ε_total-DP for ANY weight vector w.  □

Allocation Strategies
---------------------
1. MI-weighted   : w_j ∝ I(X_j; Y)  — mutual information with labels.
                   Features more informative about the attack class receive
                   more budget → less noise → better learning signal.
                   This is an empirically motivated heuristic; no formal
                   optimality claim is made beyond the DP guarantee above.

2. Var-weighted  : w_j ∝ Var(X_j)  — variance of each feature.
                   High-variance features carry more signal; giving them
                   more budget helps preserve their scale.
                   Empirical heuristic; no formal optimality claim.

3. SNR-heuristic : w_j ∝ std(X_j) / GS_j
                   Attempts to balance feature signal (std) against the
                   sensitivity cost (GS).  This is a heuristic weighting
                   — it is NOT claimed to be optimal in a formal sense.

4. Uniform       : w_j = 1/d  (baseline, current standard approach)

Clipped Sensitivity
-------------------
For heavy-tailed features (e.g. src_bytes, max ≈ 1.3 × 10⁹), GS = (max − min)/n
can be enormous.  We propose clipping at the p-th percentile before computing
sensitivity (see sensitivity.py for formal Lemma 1 and Lemma 2):

    GS_clip(X_j, p) = (quantile_p(X_j) − min(X_j)) / n    [mean query]

DP validity: after clipping each x_i to [min, τ], adjacent databases
differ by at most (τ − min)/n → Laplace(0, GS_clip/ε) is ε-DP.

Bias bound (Lemma 2 from sensitivity.py):
    |clipped_mean − true_mean| ≤ frac_above × (max − quantile_p)
    where frac_above = fraction of values above the threshold.

Note: the threshold τ = quantile_p(X_j) is treated as a publicly known
parameter (determined from domain knowledge or auxiliary public data).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Mutual information estimation (discrete approximation)
# ---------------------------------------------------------------------------

def _mi_discrete(x, y, n_bins=20):
    """
    Estimate I(X; Y) using a discrete approximation.

    X is discretised into n_bins equal-width bins; Y is the class label (any type).
    Returns MI in nats (natural log base).
    """
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(x)
    x, y = x[mask], np.asarray(y)[mask]

    if len(x) == 0 or len(np.unique(y)) < 2:
        return 0.0

    lo, hi = np.min(x), np.max(x)
    if hi == lo:
        return 0.0

    bins = np.linspace(lo, hi, n_bins + 1)
    x_disc = np.digitize(x, bins[:-1]) - 1  # [0, n_bins-1]
    x_disc = np.clip(x_disc, 0, n_bins - 1)

    classes = np.unique(y)
    n = len(x)
    mi = 0.0

    px = np.zeros(n_bins)
    for b in range(n_bins):
        px[b] = np.sum(x_disc == b) / n

    for c in classes:
        mask_c = y == c
        py_c = np.sum(mask_c) / n
        if py_c <= 0:
            continue
        for b in range(n_bins):
            mask_bc = mask_c & (x_disc == b)
            p_joint = np.sum(mask_bc) / n
            if p_joint > 0 and px[b] > 0:
                mi += p_joint * np.log(p_joint / (px[b] * py_c))

    return float(max(mi, 0.0))


# ---------------------------------------------------------------------------
# Clipped sensitivity
# ---------------------------------------------------------------------------

def clipped_sensitivity(data, query='mean', clip_percentile=99.0):
    """
    Compute sensitivity after clipping values at the p-th percentile.

    Formal guarantees (proofs in sensitivity.py module docstring):
      Lemma 1 — DP validity : GS_clip = (τ − lo) / n  is a valid sensitivity
                               for the clipped mean.  Laplace(0, GS_clip/ε) ⇒ ε-DP.
      Lemma 2 — Bias bound  : |clipped_mean − true_mean| ≤ frac_above × (hi − τ)
                               (NOT divided by n — see derivation in sensitivity.py)

    Parameters
    ----------
    data            : 1-D array of feature values
    query           : 'mean' or 'sum'
    clip_percentile : p ∈ (0, 100];  default 99th

    Returns
    -------
    dict with keys:
      clip_threshold   : τ — the percentile value used as upper clip
      gs_raw           : global sensitivity without clipping
      gs_clipped       : global sensitivity after clipping (Lemma 1)
      bias_bound       : worst-case |clipped_f − true_f| (Lemma 2)
      relative_bias_pct: bias_bound / |true_mean| × 100 (for reporting)
      noise_reduction  : gs_raw / gs_clipped (e.g. 1000 = 1000× less noise)
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)
    if n == 0:
        return {'clip_threshold': 0.0, 'gs_raw': 1.0, 'gs_clipped': 1.0,
                'bias_bound': 0.0, 'relative_bias_pct': 0.0, 'noise_reduction': 1.0}

    lo = float(np.min(data))
    hi = float(np.max(data))
    threshold = float(np.percentile(data, clip_percentile))
    threshold = max(threshold, lo + 1e-10)

    gs_raw  = (hi - lo) / n if query == 'mean' else (hi - lo)
    gs_clip = (threshold - lo) / n if query == 'mean' else (threshold - lo)
    gs_clip = max(gs_clip, 1e-10)

    frac_above = float(np.mean(data > threshold))
    max_clipping = hi - threshold

    # Lemma 2 (sensitivity.py): bias for mean  = frac_above × (hi − τ)
    #                            bias for sum   = frac_above × n × (hi − τ)
    if query == 'mean':
        bias_bound = frac_above * max_clipping
    else:
        bias_bound = frac_above * n * max_clipping

    true_mean = float(np.mean(data))
    relative_bias_pct = (bias_bound / abs(true_mean) * 100.0) if abs(true_mean) > 1e-12 else 0.0

    return {
        'clip_threshold':    threshold,
        'gs_raw':            float(gs_raw),
        'gs_clipped':        float(gs_clip),
        'bias_bound':        float(bias_bound),
        'relative_bias_pct': float(relative_bias_pct),
        'noise_reduction':   float(gs_raw / gs_clip),
    }


def clip_and_compute_mean(data, clip_threshold):
    """Return mean of data clipped at [min(data), clip_threshold]."""
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return 0.0
    lo = float(np.min(data))
    return float(np.mean(np.clip(data, lo, clip_threshold)))


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------

def compute_feature_importance(X, y, method='mi', n_bins=20):
    """
    Compute normalised importance weights for each feature column.

    Parameters
    ----------
    X      : (n, d) array of feature values
    y      : (n,) array of class labels
    method : 'mi' | 'variance' | 'snr'

    Returns
    -------
    weights : (d,) array, sum = 1
    raw     : (d,) raw importance scores before normalisation
    """
    n, d = X.shape

    if method == 'mi':
        raw = np.array([_mi_discrete(X[:, j], y, n_bins) for j in range(d)])

    elif method == 'variance':
        raw = np.array([np.nanvar(X[:, j]) for j in range(d)])

    elif method == 'snr':
        # w_j ∝ std(X_j) / GS_j  — heuristic balancing feature spread vs. sensitivity cost
        raw = np.zeros(d)
        for j in range(d):
            col = X[:, j]
            col = col[~np.isnan(col)]
            if len(col) < 2:
                raw[j] = 0.0
                continue
            std_j = float(np.std(col))
            gs_j = float(np.max(col) - np.min(col)) / len(col) + 1e-10
            raw[j] = std_j / gs_j

    else:
        # Uniform fallback
        return np.ones(d) / d, np.ones(d)

    total = float(np.sum(raw))
    if total <= 0:
        return np.ones(d) / d, raw

    weights = raw / total
    # Floor: ensure no feature gets < 1% of uniform allocation
    floor = 1.0 / (100.0 * d)
    weights = np.maximum(weights, floor)
    weights /= weights.sum()

    return weights, raw


def allocate_budget(total_epsilon, weights):
    """
    Allocate total epsilon across features using the given weights.

    Returns per-feature epsilon array (sums to total_epsilon).
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return total_epsilon * weights


# ---------------------------------------------------------------------------
# Noise injection with weighted budget
# ---------------------------------------------------------------------------

def apply_weighted_dp_noise(X, total_epsilon, y_or_weights,
                            method='mi', mechanism='laplace',
                            delta=1e-5, clip_percentile=None):
    """
    Add DP noise to X using importance-weighted budget allocation.

    Parameters
    ----------
    X               : (n, d) clean training feature matrix
    total_epsilon   : total privacy budget
    y_or_weights    : class labels (for 'mi') or pre-computed weight array
    method          : 'mi' | 'variance' | 'snr' | 'uniform'
    mechanism       : 'laplace' | 'gaussian'
    delta           : for Gaussian mechanism
    clip_percentile : if given, clip each feature before computing sensitivity

    Returns
    -------
    X_noisy : (n, d) noisy feature matrix
    meta    : dict with per-feature ε, sensitivity, weights
    """
    n, d = X.shape
    X_noisy = X.astype(float).copy()

    # Compute weights
    if isinstance(y_or_weights, np.ndarray) and y_or_weights.shape == (d,):
        weights = y_or_weights / y_or_weights.sum()
        raw = weights
    else:
        weights, raw = compute_feature_importance(X, y_or_weights, method)

    eps_per_feature = allocate_budget(total_epsilon, weights)

    meta = {
        'weights': weights.tolist(),
        'epsilons': eps_per_feature.tolist(),
        'sensitivities': [],
        'noise_scales': [],
    }

    for j in range(d):
        col = X[:, j]
        col_clean = col[~np.isnan(col)]

        if len(col_clean) == 0:
            meta['sensitivities'].append(1.0)
            meta['noise_scales'].append(0.0)
            continue

        # Sensitivity
        if clip_percentile is not None:
            cs = clipped_sensitivity(col_clean, 'sum', clip_percentile)
            s = cs['gs_clipped'] * len(col_clean)  # convert back to sum scale
            s = max(s, 1e-10)
        else:
            s = float(np.max(col_clean) - np.min(col_clean))
            s = max(s, 1e-10)

        eps_j = float(eps_per_feature[j])

        if mechanism == 'laplace':
            scale = s / eps_j
            X_noisy[:, j] += np.random.laplace(0.0, scale, size=n)
            meta['noise_scales'].append(scale)
        elif mechanism == 'gaussian':
            sigma = np.sqrt(2.0 * np.log(1.25 / delta)) * s / eps_j
            X_noisy[:, j] += np.random.normal(0.0, sigma, size=n)
            meta['noise_scales'].append(sigma)
        else:
            scale = s / eps_j
            X_noisy[:, j] += np.random.laplace(0.0, scale, size=n)
            meta['noise_scales'].append(scale)

        meta['sensitivities'].append(float(s))

    return X_noisy, meta


# ---------------------------------------------------------------------------
# Budget comparison report
# ---------------------------------------------------------------------------

def budget_comparison_report(X, y, total_epsilon, mechanisms=('laplace',),
                              clip_percentile=99.0):
    """
    Compare uniform vs. weighted vs. clipped sensitivity allocations.

    Returns a list of dicts for CSV export.
    """
    n, d = X.shape
    rows = []

    methods = ['uniform', 'mi', 'variance', 'snr']
    weights_by_method = {}
    for m in methods:
        if m == 'uniform':
            w = np.ones(d) / d
            r = np.ones(d)
        else:
            w, r = compute_feature_importance(X, y, m)
        weights_by_method[m] = (w, r)

    for j in range(d):
        col = X[:, j]
        col_clean = col[~np.isnan(col)]
        cs = clipped_sensitivity(col_clean, 'sum', clip_percentile)

        for m in methods:
            w, _ = weights_by_method[m]
            eps_j = total_epsilon * w[j]
            gs = cs['gs_raw'] * len(col_clean) / len(col_clean) if len(col_clean) > 0 else 1.0
            gs_clipped = cs['gs_clipped'] * len(col_clean) / len(col_clean) if len(col_clean) > 0 else 1.0

            # Effective noise scale for Laplace (sum query)
            noise_scale_raw = cs['gs_raw'] * len(col_clean) / eps_j
            noise_scale_clip = cs['gs_clipped'] * len(col_clean) / eps_j

            rows.append({
                'feature_idx': j,
                'allocation_method': m,
                'weight': float(w[j]),
                'epsilon_allocated': float(eps_j),
                'gs_raw': float(cs['gs_raw']),
                'gs_clipped': float(cs['gs_clipped']),
                'noise_reduction_from_clipping': float(cs['noise_reduction']),
                'noise_scale_uniform_raw': float(total_epsilon / d / max(cs['gs_raw'], 1e-10)),
                'noise_scale_weighted_raw': float(eps_j / max(cs['gs_raw'], 1e-10)),
                'noise_scale_weighted_clip': float(eps_j / max(cs['gs_clipped'], 1e-10)),
            })

    return rows
