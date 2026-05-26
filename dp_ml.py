"""
Differentially Private Machine Learning for Intrusion Detection.

Trains a binary classifier (normal vs. attack) on DP-noised training data
and evaluates on a clean held-out test set.  Sweeping epsilon shows how
classification utility degrades as privacy strengthens.

Key design choices:
  - Per-feature data-driven sensitivity (not a fixed global value)
  - Budget split equally across features (basic composition)
  - Custom numpy logistic regression — no sklearn dependency
  - Stratified train/test split
  - n_runs repetitions for confidence intervals

Privacy model: central DP (trusted aggregator adds noise to training data
before passing to the learning algorithm).
"""
import numpy as np


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_binary_labels(y_raw):
    """Encode NSL-KDD labels: 'normal' → 0, everything else → 1 (attack)."""
    y = np.asarray(y_raw, dtype=str)
    return (y != 'normal').astype(int)


# ---------------------------------------------------------------------------
# Sensitivity calibration
# ---------------------------------------------------------------------------

def compute_feature_sensitivities(X, method='range'):
    """
    Per-feature global sensitivity from training data.

    'range'  : GS = max - min  (conservative, correct for sum queries)
    'std'    : GS = 2 * std    (less conservative heuristic)
    """
    sens = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            sens.append(1.0)
            continue
        if method == 'range':
            s = float(np.max(col) - np.min(col))
        else:
            s = float(2.0 * np.std(col))
        sens.append(max(s, 1e-8))
    return np.array(sens)


# ---------------------------------------------------------------------------
# DP noise injection
# ---------------------------------------------------------------------------

def apply_dp_noise(X, total_epsilon, sensitivities, mechanism='laplace', delta=1e-5):
    """
    Add DP noise to each feature column using basic composition.

    Budget: ε_per_feature = total_ε / d  (uniform split across d features).
    Calibration uses per-feature global sensitivity.
    """
    X_noisy = X.astype(float).copy()
    d = X.shape[1]
    eps_j = total_epsilon / d
    for j in range(d):
        s = sensitivities[j]
        if mechanism == 'laplace':
            scale = s / eps_j
            X_noisy[:, j] += np.random.laplace(0, scale, size=X.shape[0])
        elif mechanism == 'gaussian':
            sigma = np.sqrt(2.0 * np.log(1.25 / delta)) * s / eps_j
            X_noisy[:, j] += np.random.normal(0, sigma, size=X.shape[0])
        else:
            scale = s / eps_j
            X_noisy[:, j] += np.random.laplace(0, scale, size=X.shape[0])
    return X_noisy


# ---------------------------------------------------------------------------
# Logistic regression (pure numpy, no sklearn)
# ---------------------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _standardize(X_tr, X_te):
    mu = np.mean(X_tr, axis=0)
    sd = np.std(X_tr, axis=0) + 1e-8
    return (X_tr - mu) / sd, (X_te - mu) / sd


class _LogisticRegression:
    """Gradient-descent logistic regression (binary, L2 regularised)."""

    def __init__(self, lr=0.3, n_iter=300, lam=0.01):
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


# ---------------------------------------------------------------------------
# Metrics (no sklearn)
# ---------------------------------------------------------------------------

def _accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def _f1(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_dp_ml_experiment(X_train, X_test, y_train, y_test,
                          epsilons, mechanism='laplace',
                          n_runs=5, delta=1e-5):
    """
    Sweep epsilon and measure classification accuracy/F1 on NSL-KDD.

    Protocol
    --------
    1. Compute per-feature sensitivities from clean training data.
    2. For each ε: add DP noise to X_train, train LR, evaluate on X_test.
    3. Repeat n_runs times; report mean ± std.
    4. Compare against a clean (non-private) baseline.

    Returns
    -------
    (results_list, baseline_accuracy)
    """
    sensitivities = compute_feature_sensitivities(X_train)

    # Non-private baseline
    X_tr_s, X_te_s = _standardize(X_train, X_test)
    clf0 = _LogisticRegression()
    clf0.fit(X_tr_s, y_train)
    y_pred0 = clf0.predict(X_te_s)
    base_acc = _accuracy(y_test, y_pred0)
    base_f1 = _f1(y_test, y_pred0)

    results = [{
        'epsilon': float('inf'),
        'mechanism': 'none',
        'accuracy': base_acc,
        'f1_score': base_f1,
        'accuracy_std': 0.0,
        'f1_std': 0.0,
        'relative_accuracy_loss_pct': 0.0,
        'privacy_type': 'no_dp',
    }]

    for eps in epsilons:
        accs, f1s = [], []
        for _ in range(n_runs):
            X_tr_noisy = apply_dp_noise(X_train, eps, sensitivities, mechanism, delta)
            X_tr_ns, X_te_ns = _standardize(X_tr_noisy, X_test)
            clf = _LogisticRegression()
            clf.fit(X_tr_ns, y_train)
            y_pred = clf.predict(X_te_ns)
            accs.append(_accuracy(y_test, y_pred))
            f1s.append(_f1(y_test, y_pred))

        mean_acc = float(np.mean(accs))
        results.append({
            'epsilon': float(eps),
            'mechanism': mechanism,
            'accuracy': mean_acc,
            'f1_score': float(np.mean(f1s)),
            'accuracy_std': float(np.std(accs)),
            'f1_std': float(np.std(f1s)),
            'relative_accuracy_loss_pct': (base_acc - mean_acc) / base_acc * 100.0,
            'privacy_type': 'central_dp',
        })

    return results, base_acc


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_kdd_data(df_pandas, numerical_features, label_col,
                     test_size=0.3, seed=42, max_rows=None):
    """
    Prepare NSL-KDD data for binary classification.

    Parameters
    ----------
    df_pandas        : pandas DataFrame with the full dataset
    numerical_features: list of column names to use as features
    label_col        : column containing attack labels ('normal' / attack names)
    test_size        : fraction for test set
    max_rows         : subsample to this many rows (for speed)

    Returns
    -------
    X_train, X_test, y_train, y_test  (numpy arrays)
    """
    available = [c for c in numerical_features if c in df_pandas.columns]
    X = df_pandas[available].values.astype(float)

    if label_col and label_col in df_pandas.columns:
        y = encode_binary_labels(df_pandas[label_col].values)
    else:
        y = np.zeros(len(X), dtype=int)

    # Drop NaN rows
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    # Optional subsample for speed
    if max_rows and len(X) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X[idx], y[idx]

    # Stratified split
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_test = int(len(pos_idx) * test_size)
    n_neg_test = int(len(neg_idx) * test_size)

    test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
    train_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
