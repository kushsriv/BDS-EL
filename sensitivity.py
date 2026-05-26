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


def sensitivity_report(data, query='mean'):
    """Return a summary dict of GS, LS and SS for a feature."""
    gs = compute_global_sensitivity(data, query)
    ls = compute_local_sensitivity(data, query)
    beta = 0.01
    ss = compute_smooth_sensitivity(data, beta, query)
    return {
        'global_sensitivity': gs,
        'local_sensitivity': ls,
        f'smooth_sensitivity_beta{beta}': ss,
        'ls_gs_ratio': ls / gs if gs > 0 else 0.0,
    }
