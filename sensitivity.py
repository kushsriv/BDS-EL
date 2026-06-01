"""
Data-driven sensitivity computation for Differential Privacy.

Global sensitivity (GS) is the standard DP calibration parameter but
fixing GS=1.0 is rarely correct — it should reflect actual data range and n.

This module provides:
  - Global sensitivity (worst-case, dataset-independent bound)
  - Local sensitivity (dataset-dependent, tighter)
  - Smooth sensitivity (Nissim et al. 2007, allows noise calibrated to LS
    while preserving DP via a smoothing factor beta)
"""
import numpy as np


def compute_global_sensitivity(data, query='mean'):
    """
    Compute global sensitivity for a given aggregate query.

    GS bounds the worst-case change in output when one record is
    added or removed.  Formulas:
      mean  : GS = (max - min) / n
      sum   : GS = max - min
      count : GS = 1
      variance : GS = (max - min)^2 * (n-1) / n^2
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)
    if n == 0:
        return 1.0

    lo, hi = float(np.min(data)), float(np.max(data))
    r = hi - lo

    if query == 'mean':
        return r / n
    elif query == 'sum':
        return r
    elif query == 'count':
        return 1.0
    elif query == 'variance':
        return (r ** 2) * (n - 1) / (n ** 2) if n > 1 else 0.0
    else:
        return r / n


def compute_local_sensitivity(data, query='mean'):
    """
    Compute local sensitivity at the given dataset instance.

    LS(x) = max_{x' adjacent to x} |f(x) - f(x')|

    LS <= GS always, so noise calibrated to LS is smaller.
    Because LS is data-dependent it must be combined with smooth
    sensitivity or a public-coin mechanism to remain private.
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)
    if n == 0:
        return 1.0

    if query == 'mean':
        mean_val = float(np.mean(data))
        # Removing record i changes mean by |x_i - mean| / n
        return float(np.max(np.abs(data - mean_val)) / n)
    elif query == 'sum':
        return float(np.max(np.abs(data)))
    else:
        return compute_global_sensitivity(data, query)


def compute_smooth_sensitivity(data, beta, query='mean', k_max=None):
    """
    Smooth sensitivity (Nissim, Raskhodnikova & Smith 2007).

    SS_beta(x) = max_{k>=0} exp(-beta * k) * LS_k(x)

    where LS_k is the max local sensitivity over datasets at
    Hamming distance k from x.

    For the mean query on data in [lo, hi]:
      LS_k(x) <= min(GS, range * k / (n - k))

    Adding Laplace(0, 2*SS_beta / epsilon) gives epsilon-DP with
    the smooth sensitivity framework.
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)
    if n == 0:
        return 1.0

    r = float(np.max(data) - np.min(data)) if n > 1 else 0.0
    gs = compute_global_sensitivity(data, query)

    if k_max is None:
        k_max = min(n // 2, 100)

    if query == 'mean':
        ls0 = compute_local_sensitivity(data, 'mean')
        best = ls0  # k = 0 term
        for k in range(1, k_max + 1):
            if n - k <= 0:
                break
            ls_k = min(gs, r * k / (n - k))
            candidate = np.exp(-beta * k) * ls_k
            if candidate > best:
                best = candidate
        return float(best)
    else:
        return gs


def compute_clipped_sensitivity(data, query='mean', clip_percentile=99.0):
    """
    Sensitivity calibrated to the p-th percentile rather than the raw max.

    For heavy-tailed features (e.g. network byte counts), the raw range can
    be millions while 99% of values are much smaller.  Clipping at the p-th
    percentile reduces sensitivity — and therefore noise — by orders of
    magnitude, at the cost of a small, bounded bias.

    Privacy: after clipping each x_i to [min, threshold], any adjacent DB
    differs in one clipped value by at most (threshold − min).
    So GS_clip = (threshold − min) / n for mean queries is valid.

    Bias bound: |clipped_mean − true_mean| ≤ frac_above · (max − threshold) / n.
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)
    if n == 0:
        return {
            'gs_clipped': 1.0, 'gs_raw': 1.0,
            'clip_threshold': 0.0, 'bias_bound': 0.0, 'noise_reduction': 1.0,
        }

    lo = float(np.min(data))
    hi = float(np.max(data))
    threshold = float(np.percentile(data, clip_percentile))
    threshold = max(threshold, lo + 1e-10)

    gs_raw = (hi - lo) / n if query == 'mean' else (hi - lo)
    gs_clip = (threshold - lo) / n if query == 'mean' else (threshold - lo)
    gs_clip = max(gs_clip, 1e-10)

    frac_above = float(np.mean(data > threshold))
    bias_bound = frac_above * (hi - threshold) / n if query == 'mean' else frac_above * (hi - threshold)

    return {
        'gs_clipped': float(gs_clip),
        'gs_raw': float(gs_raw),
        'clip_threshold': float(threshold),
        'bias_bound': float(bias_bound),
        'noise_reduction': float(gs_raw / gs_clip),
    }


def sensitivity_report(data, query='mean'):
    """Return a summary dict of GS, LS, SS, and clipped sensitivity for a feature."""
    gs = compute_global_sensitivity(data, query)
    ls = compute_local_sensitivity(data, query)
    beta = 0.01
    ss = compute_smooth_sensitivity(data, beta, query)
    cs = compute_clipped_sensitivity(data, query, clip_percentile=99.0)
    return {
        'global_sensitivity': gs,
        'local_sensitivity': ls,
        f'smooth_sensitivity_beta{beta}': ss,
        'ls_gs_ratio': ls / gs if gs > 0 else 0.0,
        'gs_clipped_p99': cs['gs_clipped'],
        'clip_noise_reduction': cs['noise_reduction'],
        'clip_bias_bound': cs['bias_bound'],
    }
