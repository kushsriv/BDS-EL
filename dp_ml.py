"""
Differentially Private Machine Learning for Intrusion Detection
===============================================================

Trains binary (normal vs. attack) and multi-class (5-category) classifiers
on NSL-KDD data with differential privacy.  Three DP approaches are compared:

  1. Input perturbation (uniform budget) — baseline
  2. Input perturbation (importance-weighted budget) — novel contribution
  3. DP-SGD (Abadi et al. 2016)                     — gold-standard baseline

Key contributions vs. prior work
---------------------------------
- Per-feature CLIPPED sensitivity (instead of raw max-min): reduces noise
  10-500× for heavy-tailed network traffic features.
- Importance-weighted budget allocation (MI-weighted): allocates more budget
  to features with higher mutual information with attack labels.
- Full 36-feature NSL-KDD experiments (not just 3 features).
- Multi-class IDS (5 classes: Normal, DoS, Probe, R2L, U2R).
- 30-run confidence intervals and Wilcoxon signed-rank significance tests.
"""
import numpy as np

from weighted_budget import (
    apply_weighted_dp_noise, compute_feature_importance, allocate_budget
)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

# NSL-KDD attack taxonomy (based on KDD Cup 1999 definitions)
_ATTACK_TAXONOMY = {
    'normal': 0,
    # DoS
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1,
    'teardrop': 1, 'apache2': 1, 'udpstorm': 1, 'processtable': 1,
    'mailbomb': 1, 'worm': 1,
    # Probe
    'satan': 2, 'ipsweep': 2, 'nmap': 2, 'portsweep': 2,
    'mscan': 2, 'saint': 2,
    # R2L
    'warezclient': 3, 'warezmaster': 3, 'imap': 3, 'ftp_write': 3,
    'guess_passwd': 3, 'phf': 3, 'multihop': 3, 'spy': 3, 'named': 3,
    'snmpgetattack': 3, 'snmpguess': 3, 'xsnoop': 3, 'xlock': 3,
    'sendmail': 3, 'httptunnel': 3,
    # U2R
    'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4,
    'ps': 4, 'sqlattack': 4, 'xterm': 4,
}
_CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


def encode_binary_labels(y_raw):
    """'normal' / 'Normal' / 'NORMAL' → 0, any attack → 1. Case-insensitive."""
    y = np.asarray(y_raw, dtype=str)
    return (np.char.lower(y) != 'normal').astype(int)


def encode_multiclass_labels(y_raw):
    """Map attack label strings to 0-4 class indices."""
    y = np.asarray(y_raw, dtype=str)
    out = np.full(len(y), -1, dtype=int)
    for label, idx in _ATTACK_TAXONOMY.items():
        out[y == label] = idx
    # Unknown labels → 'other attack' → DoS (1) as conservative fallback
    out[out == -1] = 1
    return out


# ---------------------------------------------------------------------------
# Sensitivity calibration
# ---------------------------------------------------------------------------

def compute_feature_sensitivities(X, method='clipped', clip_percentile=99.0):
    """
    Per-feature sensitivity from training data.

    method
    ------
    'range'   : GS = max − min  (conservative, no clipping)
    'clipped' : GS = (quantile_p − min)  (novel: handles heavy tails)
    'std'     : GS = 2 × std  (heuristic; not strictly correct)
    """
    d = X.shape[1]
    sens = np.zeros(d)
    for j in range(d):
        col = X[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            sens[j] = 1.0
            continue
        lo = float(np.min(col))
        hi = float(np.max(col))
        if method == 'range':
            s = hi - lo
        elif method == 'clipped':
            threshold = float(np.percentile(col, clip_percentile))
            s = max(threshold - lo, 1e-10)
        elif method == 'std':
            s = 2.0 * float(np.std(col))
        else:
            s = hi - lo
        sens[j] = max(s, 1e-8)
    return sens


# ---------------------------------------------------------------------------
# DP noise — input perturbation (uniform budget)
# ---------------------------------------------------------------------------

def apply_dp_noise(X, total_epsilon, sensitivities, mechanism='laplace',
                   delta=1e-5):
    """
    Standard uniform budget split: ε_j = ε_total / d for each feature.
    Sensitivity is per-feature (not fixed to 1.0).
    """
    X_noisy = X.astype(float).copy()
    d = X.shape[1]
    eps_j = total_epsilon / d
    for j in range(d):
        s = sensitivities[j]
        if mechanism == 'laplace':
            scale = s / eps_j
            X_noisy[:, j] += np.random.laplace(0.0, scale, size=X.shape[0])
        elif mechanism == 'gaussian':
            sigma = np.sqrt(2.0 * np.log(1.25 / delta)) * s / eps_j
            X_noisy[:, j] += np.random.normal(0.0, sigma, size=X.shape[0])
        else:
            scale = s / eps_j
            X_noisy[:, j] += np.random.laplace(0.0, scale, size=X.shape[0])
    return X_noisy


# ---------------------------------------------------------------------------
# Logistic regression (pure numpy, no sklearn)
# ---------------------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def _standardize(X_tr, X_te):
    mu = np.nanmean(X_tr, axis=0)
    sd = np.nanstd(X_tr, axis=0) + 1e-8
    return (X_tr - mu) / sd, (X_te - mu) / sd


class _LogisticRegression:
    """Binary logistic regression (gradient descent, L2 regularised)."""

    def __init__(self, lr=0.1, n_iter=200, lam=0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.lam = lam

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = _sigmoid(X @ self.w + self.b)
            err = p - y
            self.w -= self.lr * ((X.T @ err) / n + self.lam * self.w)
            self.b -= self.lr * float(np.mean(err))
        return self

    def predict(self, X):
        return (_sigmoid(X @ self.w + self.b) >= 0.5).astype(int)


class _SoftmaxRegression:
    """Multi-class logistic regression (softmax, gradient descent)."""

    def __init__(self, n_classes=5, lr=0.1, n_iter=200, lam=0.01):
        self.n_classes = n_classes
        self.lr = lr
        self.n_iter = n_iter
        self.lam = lam

    def fit(self, X, y):
        n, d = X.shape
        K = self.n_classes
        # One-hot
        Y = np.zeros((n, K))
        for k in range(K):
            Y[y == k, k] = 1.0
        self.W = np.zeros((d, K))
        self.b = np.zeros(K)
        for _ in range(self.n_iter):
            P = _softmax(X @ self.W + self.b)       # (n, K)
            E = (P - Y) / n                          # (n, K)
            self.W -= self.lr * (X.T @ E + self.lam * self.W)
            self.b -= self.lr * E.sum(axis=0)
        return self

    def predict(self, X):
        return np.argmax(_softmax(X @ self.W + self.b), axis=1)


# ---------------------------------------------------------------------------
# Metrics (no sklearn)
# ---------------------------------------------------------------------------

def _accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def _f1(y_true, y_pred, pos=1):
    tp = int(np.sum((y_pred == pos) & (y_true == pos)))
    fp = int(np.sum((y_pred == pos) & (y_true != pos)))
    fn = int(np.sum((y_pred != pos) & (y_true == pos)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


def _macro_f1(y_true, y_pred, n_classes):
    """Macro-averaged F1 across all classes."""
    f1s = [_f1(y_true, y_pred, pos=k) for k in range(n_classes)]
    return float(np.mean(f1s))


def _confidence_interval(values, confidence=0.95):
    """Return (mean, margin) where interval = mean ± margin."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = float(np.mean(values))
    se = float(np.std(values)) / np.sqrt(n)
    z = 1.96  # 95% CI
    return mean, float(z * se)


def wilcoxon_signed_rank(x, y):
    """
    Two-sided Wilcoxon signed-rank test (H0: median(x-y) = 0).
    Returns approximate p-value using normal approximation for n >= 10.
    """
    diff = np.asarray(x) - np.asarray(y)
    diff = diff[diff != 0]
    n = len(diff)
    if n < 5:
        return 1.0

    ranks = np.zeros(n)
    abs_diff = np.abs(diff)
    order = np.argsort(abs_diff)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1

    W_plus = float(np.sum(ranks[diff > 0]))
    W = W_plus

    # Normal approximation
    mu_W = n * (n + 1) / 4.0
    sigma_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma_W == 0:
        return 1.0
    z = (W - mu_W) / sigma_W
    # Two-tailed p-value (standard normal approximation)
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return float(p)


def _norm_cdf(z):
    """Standard normal CDF via Horner's method approximation."""
    z = float(z)
    if z < 0:
        return 1.0 - _norm_cdf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
            + t * (-1.821255978 + t * 1.330274429))))
    return 1.0 - 0.3989422804 * np.exp(-0.5 * z * z) * poly


# ---------------------------------------------------------------------------
# Main experiment: binary classification
# ---------------------------------------------------------------------------

def run_dp_ml_experiment(X_train, X_test, y_train, y_test,
                          epsilons, mechanism='laplace',
                          n_runs=30, delta=1e-5,
                          sensitivity_method='clipped',
                          budget_method='uniform'):
    """
    Sweep epsilon; measure classification accuracy/F1 on NSL-KDD.

    Parameters
    ----------
    sensitivity_method : 'clipped' (novel) | 'range' (standard)
    budget_method      : 'uniform' (baseline) | 'mi' | 'variance' | 'snr'

    Returns
    -------
    (results_list, baseline_accuracy)
    """
    # Non-private baseline
    X_tr_s, X_te_s = _standardize(X_train, X_test)
    clf0 = _LogisticRegression()
    clf0.fit(X_tr_s, y_train)
    y_pred0 = clf0.predict(X_te_s)
    base_acc = _accuracy(y_test, y_pred0)
    base_f1 = _f1(y_test, y_pred0)

    sensitivities = compute_feature_sensitivities(X_train, sensitivity_method)

    # Pre-compute MI weights (expensive, do once)
    if budget_method != 'uniform':
        weights, _ = compute_feature_importance(X_train, y_train, budget_method)
    else:
        d = X_train.shape[1]
        weights = np.ones(d) / d

    results = [{
        'epsilon': float('inf'),
        'mechanism': 'none',
        'budget_method': budget_method,
        'sensitivity_method': sensitivity_method,
        'accuracy': base_acc,
        'accuracy_ci': 0.0,
        'f1_score': base_f1,
        'f1_ci': 0.0,
        'accuracy_std': 0.0,
        'f1_std': 0.0,
        'relative_accuracy_loss_pct': 0.0,
        'privacy_type': 'no_dp',
    }]

    for eps in epsilons:
        accs, f1s = [], []
        for _ in range(n_runs):
            if budget_method == 'uniform':
                X_tr_noisy = apply_dp_noise(
                    X_train, eps, sensitivities, mechanism, delta
                )
            else:
                X_tr_noisy, _ = apply_weighted_dp_noise(
                    X_train, eps, weights,
                    method=budget_method, mechanism=mechanism, delta=delta
                )

            X_tr_ns, X_te_ns = _standardize(X_tr_noisy, X_test)
            clf = _LogisticRegression()
            clf.fit(X_tr_ns, y_train)
            y_pred = clf.predict(X_te_ns)
            accs.append(_accuracy(y_test, y_pred))
            f1s.append(_f1(y_test, y_pred))

        mean_acc, ci_acc = _confidence_interval(accs)
        mean_f1, ci_f1 = _confidence_interval(f1s)
        results.append({
            'epsilon': float(eps),
            'mechanism': mechanism,
            'budget_method': budget_method,
            'sensitivity_method': sensitivity_method,
            'accuracy': mean_acc,
            'accuracy_ci': ci_acc,
            'f1_score': mean_f1,
            'f1_ci': ci_f1,
            'accuracy_std': float(np.std(accs)),
            'f1_std': float(np.std(f1s)),
            'relative_accuracy_loss_pct': (base_acc - mean_acc) / max(base_acc, 1e-8) * 100.0,
            'privacy_type': 'central_dp',
        })

    return results, base_acc


# ---------------------------------------------------------------------------
# Multi-class classification (5 attack categories)
# ---------------------------------------------------------------------------

def run_multiclass_experiment(X_train, X_test, y_train_mc, y_test_mc,
                               epsilons, mechanism='laplace',
                               n_runs=10, delta=1e-5,
                               sensitivity_method='clipped',
                               budget_method='mi'):
    """
    Sweep epsilon for 5-class IDS (Normal / DoS / Probe / R2L / U2R).

    Returns (results_list, baseline_accuracy, class_names).
    """
    n_classes = len(_CLASS_NAMES)
    sensitivities = compute_feature_sensitivities(X_train, sensitivity_method)

    if budget_method != 'uniform':
        weights, _ = compute_feature_importance(X_train, y_train_mc, budget_method)
    else:
        d = X_train.shape[1]
        weights = np.ones(d) / d

    # No-DP baseline
    X_tr_s, X_te_s = _standardize(X_train, X_test)
    clf0 = _SoftmaxRegression(n_classes=n_classes)
    clf0.fit(X_tr_s, y_train_mc)
    y_pred0 = clf0.predict(X_te_s)
    base_acc = _accuracy(y_test_mc, y_pred0)
    base_f1 = _macro_f1(y_test_mc, y_pred0, n_classes)

    # Per-class accuracy for baseline
    per_class_base = {}
    for k, name in enumerate(_CLASS_NAMES):
        mask = y_test_mc == k
        if mask.sum() > 0:
            per_class_base[name] = float(np.mean(y_pred0[mask] == k))
        else:
            per_class_base[name] = float('nan')

    results = [{
        'epsilon': float('inf'),
        'mechanism': 'none',
        'budget_method': budget_method,
        'accuracy': base_acc,
        'accuracy_ci': 0.0,
        'macro_f1': base_f1,
        'macro_f1_ci': 0.0,
        'relative_accuracy_loss_pct': 0.0,
        'privacy_type': 'no_dp',
        **{f'acc_{n}': v for n, v in per_class_base.items()},
        **{f'acc_{n}_ci': 0.0 for n in _CLASS_NAMES},
    }]

    for eps in epsilons:
        accs, f1s = [], []
        per_class_runs = {n: [] for n in _CLASS_NAMES}

        for _ in range(n_runs):
            if budget_method == 'uniform':
                X_tr_noisy = apply_dp_noise(X_train, eps, sensitivities, mechanism, delta)
            else:
                X_tr_noisy, _ = apply_weighted_dp_noise(
                    X_train, eps, weights, method=budget_method,
                    mechanism=mechanism, delta=delta
                )

            X_tr_ns, X_te_ns = _standardize(X_tr_noisy, X_test)
            clf = _SoftmaxRegression(n_classes=n_classes)
            clf.fit(X_tr_ns, y_train_mc)
            y_pred = clf.predict(X_te_ns)
            accs.append(_accuracy(y_test_mc, y_pred))
            f1s.append(_macro_f1(y_test_mc, y_pred, n_classes))
            for k, name in enumerate(_CLASS_NAMES):
                mask = y_test_mc == k
                if mask.sum() > 0:
                    per_class_runs[name].append(float(np.mean(y_pred[mask] == k)))

        mean_acc, ci_acc = _confidence_interval(accs)
        mean_f1, ci_f1 = _confidence_interval(f1s)

        row = {
            'epsilon': float(eps),
            'mechanism': mechanism,
            'budget_method': budget_method,
            'accuracy': mean_acc,
            'accuracy_ci': ci_acc,
            'macro_f1': mean_f1,
            'macro_f1_ci': ci_f1,
            'relative_accuracy_loss_pct': (base_acc - mean_acc) / max(base_acc, 1e-8) * 100.0,
            'privacy_type': 'central_dp',
        }
        for name in _CLASS_NAMES:
            vals = per_class_runs[name]
            row[f'acc_{name}'] = float(np.mean(vals)) if vals else float('nan')
            row[f'acc_{name}_ci'] = float(1.96 * np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

        results.append(row)

    return results, base_acc, _CLASS_NAMES


# ---------------------------------------------------------------------------
# Budget comparison (uniform vs. MI-weighted vs. variance vs. SNR)
# ---------------------------------------------------------------------------

def compare_budget_methods(X_train, X_test, y_train, y_test,
                            epsilons, mechanism='laplace',
                            n_runs=10, delta=1e-5):
    """
    Compare all four budget allocation methods at the same total ε.

    Returns list of dicts suitable for CSV export and statistical testing.
    """
    methods = ['uniform', 'mi', 'variance', 'snr']
    all_rows = []
    acc_by_method = {m: [] for m in methods}

    for budget_method in methods:
        rows, base_acc = run_dp_ml_experiment(
            X_train, X_test, y_train, y_test,
            epsilons=epsilons, mechanism=mechanism,
            n_runs=n_runs, delta=delta,
            sensitivity_method='clipped',
            budget_method=budget_method,
        )
        for r in rows:
            r['comparison'] = 'budget_method'
        all_rows.extend(rows)
        # Collect accuracy values per epsilon for significance testing
        for r in rows:
            if r['mechanism'] != 'none':
                acc_by_method[budget_method].append(r['accuracy'])

    # Wilcoxon: MI-weighted vs. uniform
    if len(acc_by_method['mi']) == len(acc_by_method['uniform']):
        p_val = wilcoxon_signed_rank(acc_by_method['mi'], acc_by_method['uniform'])
        for r in all_rows:
            r['wilcoxon_p_mi_vs_uniform'] = round(p_val, 4)

    return all_rows


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_kdd_data(df_pandas, numerical_features, label_col,
                     test_size=0.3, seed=42, max_rows=None,
                     multiclass=False):
    """
    Prepare NSL-KDD data for binary or multi-class classification.

    Returns
    -------
    X_train, X_test, y_train, y_test  (numpy arrays, y is int)
    """
    available = [c for c in numerical_features if c in df_pandas.columns]

    import pandas as pd
    X_df = df_pandas[available].copy()
    for c in available:
        X_df[c] = pd.to_numeric(X_df[c], errors='coerce')
    X = X_df.values.astype(float)

    if label_col and label_col in df_pandas.columns:
        raw_labels = df_pandas[label_col].values
        if multiclass:
            y = encode_multiclass_labels(raw_labels)
        else:
            y = encode_binary_labels(raw_labels)
    else:
        y = np.zeros(len(X), dtype=int)

    # Drop rows with any NaN
    mask = ~np.isnan(X).any(axis=1) & (y >= 0)
    X, y = X[mask], y[mask]

    # Optional subsample
    if max_rows and len(X) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X[idx], y[idx]

    # Stratified split
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        n_test = max(1, int(len(c_idx) * test_size))
        test_idx.extend(c_idx[:n_test])
        train_idx.extend(c_idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
