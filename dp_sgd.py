"""
DP-SGD: Differentially Private Stochastic Gradient Descent
===========================================================

Implements the algorithm from Abadi et al. (2016) "Deep Learning with
Differential Privacy" (CCS 2016).

Protocol (one training step)
----------------------------
For mini-batch B of size L sampled from n records:
  1. Compute per-sample gradient g_i for each i ∈ B
  2. Clip:    g̃_i = g_i · min(1, C / ||g_i||₂)
  3. Aggregate + noise:  G = Σ g̃_i + N(0, (σ·C)² · I)
  4. Update:  θ ← θ − lr · G / L

Privacy Accounting (RDP)
------------------------
Each step applies a Gaussian mechanism with:
  - Sensitivity C / L  (per-sample gradient bound)
  - Noise multiplier σ (σ_rms = σ · C is the actual noise std)
  - Subsampling rate q = L / n  (amplifies privacy by subsampling)

We use the RDP bound for Poisson subsampling of Gaussians
(Mironov 2017, Theorem 8):

  RDP_α(step) ≈ (1/(α-1)) · log(
      (1-q)^(α-1) · [(1-q) + q·exp((α-1)·α/(2σ²))] +
      ...
  )

For a simpler closed form we use the approximation (tight for small q):
  RDP_α(step) ≈ q² · α / (2σ²)   [first-order in q]

After T steps, composition gives:
  RDP_α(T steps) = T · RDP_α(step)

Convert to (ε, δ) via:
  ε(δ) = RDP_α(T steps) + log(1/δ) / (α - 1)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _per_sample_gradient(X_batch, y_batch, w, b):
    """
    Compute per-sample gradients for logistic regression.

    Returns
    -------
    grads_w : (L, d) — gradient w.r.t. w for each sample
    grads_b : (L,)   — gradient w.r.t. b for each sample
    """
    p = _sigmoid(X_batch @ w + b)               # (L,)
    err = p - y_batch                            # (L,)
    grads_w = X_batch * err[:, np.newaxis]       # (L, d)
    grads_b = err                                # (L,)
    return grads_w, grads_b


def _clip_gradients(grads_w, grads_b, clip_norm):
    """
    Per-sample gradient clipping to L2 norm C.

    Returns clipped (grads_w, grads_b).
    """
    L, d = grads_w.shape
    norms = np.sqrt(np.sum(grads_w ** 2, axis=1) + grads_b ** 2)  # (L,)
    scale = np.minimum(1.0, clip_norm / (norms + 1e-10))          # (L,)
    grads_w_c = grads_w * scale[:, np.newaxis]
    grads_b_c = grads_b * scale
    return grads_w_c, grads_b_c


# ---------------------------------------------------------------------------
# DP-SGD privacy accounting
# ---------------------------------------------------------------------------

def rdp_gaussian_subsampled(alpha, sigma, q):
    """
    RDP of Poisson-subsampled Gaussian mechanism (Mironov 2017 / Wang 2019).

    Approximation valid for small q (q ≤ 0.1) and σ ≥ 0.5:
        RDP_α ≈ q² · α / (2σ²)

    For q > 0.1 we use the exact formula (tensor product of Gaussian moments):
        RDP_α = (1/(α-1)) · log(
            (1-q)^(α-1) · (1-q) +
            q · (1-q)^(α-1) · exp((α-1)/σ²) +
            q² · exp(α·(α-1)/(2σ²))
        )   [simplified, ignoring higher-order cross-terms]
    """
    if alpha == 1:
        return float(q ** 2 / (2.0 * sigma ** 2))

    a, s, qq = float(alpha), float(sigma), float(q)

    if qq <= 0.1:
        return float(qq ** 2 * a / (2.0 * s ** 2))

    # Exact 3-term expansion (safe for moderate q)
    t1 = (1.0 - qq) ** (a - 1) * (1.0 - qq)
    exponent2 = (a - 1.0) / (s ** 2)
    if exponent2 < 500:
        t2 = qq * (1.0 - qq) ** (a - 1) * np.exp(exponent2)
    else:
        t2 = 0.0

    exponent3 = a * (a - 1.0) / (2.0 * s ** 2)
    if exponent3 < 500:
        t3 = (qq ** 2) * np.exp(exponent3)
    else:
        t3 = float('inf')

    val = t1 + t2 + t3
    if val <= 0 or not np.isfinite(val):
        return float(qq ** 2 * a / (2.0 * s ** 2))

    return float(np.log(val) / (a - 1.0))


def dp_sgd_privacy_spent(n_samples, batch_size, n_epochs, sigma,
                          delta=1e-5, alphas=None):
    """
    Compute (ε, δ) privacy spent by DP-SGD after training.

    Parameters
    ----------
    n_samples  : total dataset size n
    batch_size : mini-batch size L
    n_epochs   : number of training epochs
    sigma      : noise multiplier (noise std = sigma * clip_norm)
    delta      : target δ
    alphas     : list of RDP orders to search over

    Returns
    -------
    epsilon    : best ε at the given δ
    best_alpha : order α that achieves the best bound
    """
    if alphas is None:
        alphas = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0]

    q = batch_size / n_samples
    T = int(n_epochs * n_samples / batch_size)  # total number of steps

    best_eps = float('inf')
    best_alpha = None

    for alpha in alphas:
        rdp_step = rdp_gaussian_subsampled(alpha, sigma, q)
        rdp_total = T * rdp_step
        eps = rdp_total + np.log(1.0 / delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps = eps
            best_alpha = alpha

    return float(best_eps), best_alpha


# ---------------------------------------------------------------------------
# DP-SGD logistic regression
# ---------------------------------------------------------------------------

def _standardize(X_tr, X_te):
    mu = np.nanmean(X_tr, axis=0)
    sd = np.nanstd(X_tr, axis=0) + 1e-8
    X_tr_s = (X_tr - mu) / sd
    X_te_s = (X_te - mu) / sd
    return X_tr_s, X_te_s, mu, sd


class DPSGDLogisticRegression:
    """
    Logistic regression trained with DP-SGD.

    Parameters
    ----------
    clip_norm   : C — per-sample gradient L2 clipping bound
    sigma       : noise multiplier (Gaussian noise std = sigma * clip_norm)
    lr          : learning rate
    batch_size  : mini-batch size L
    n_epochs    : number of passes over the data
    lam         : L2 regularisation coefficient
    """

    def __init__(self, clip_norm=1.0, sigma=1.0, lr=0.1,
                 batch_size=256, n_epochs=30, lam=1e-4):
        self.clip_norm = clip_norm
        self.sigma = sigma
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lam = lam
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        C = self.clip_norm
        L = min(self.batch_size, n)
        n_steps_per_epoch = max(1, n // L)

        rng = np.random.default_rng()

        for epoch in range(self.n_epochs):
            idx = rng.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]

            for step in range(n_steps_per_epoch):
                start = step * L
                end = min(start + L, n)
                X_b = X_shuf[start:end]
                y_b = y_shuf[start:end]
                L_actual = len(y_b)

                # Per-sample gradients
                gw, gb = _per_sample_gradient(X_b, y_b, self.w, self.b)

                # Clip
                gw_c, gb_c = _clip_gradients(gw, gb, C)

                # Sum clipped gradients + Gaussian noise
                noise_w = np.random.normal(0.0, self.sigma * C, size=d)
                noise_b = np.random.normal(0.0, self.sigma * C)
                G_w = gw_c.sum(axis=0) + noise_w
                G_b = gb_c.sum() + noise_b

                # Update
                self.w -= self.lr * (G_w / L_actual + self.lam * self.w)
                self.b -= self.lr * G_b / L_actual

        return self

    def predict(self, X):
        return (_sigmoid(X @ self.w + self.b) >= 0.5).astype(int)

    def predict_proba(self, X):
        p = _sigmoid(X @ self.w + self.b)
        return np.stack([1 - p, p], axis=1)


# ---------------------------------------------------------------------------
# Noise multiplier search
# ---------------------------------------------------------------------------

def find_sigma_for_epsilon(target_epsilon, n_samples, batch_size, n_epochs,
                            delta=1e-5, sigma_lo=0.1, sigma_hi=100.0,
                            tol=1e-3, max_iter=50):
    """
    Binary search for the noise multiplier σ that achieves target_epsilon.

    Returns (sigma, actual_epsilon).
    """
    for _ in range(max_iter):
        sigma_mid = (sigma_lo + sigma_hi) / 2.0
        eps, _ = dp_sgd_privacy_spent(n_samples, batch_size, n_epochs,
                                       sigma_mid, delta)
        if abs(eps - target_epsilon) < tol:
            break
        if eps > target_epsilon:
            sigma_lo = sigma_mid  # need more noise
        else:
            sigma_hi = sigma_mid  # have more than enough noise

    actual_eps, _ = dp_sgd_privacy_spent(n_samples, batch_size, n_epochs,
                                          sigma_mid, delta)
    return float(sigma_mid), float(actual_eps)


# ---------------------------------------------------------------------------
# Experiment runner
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


def run_dpsgd_experiment(X_train, X_test, y_train, y_test, epsilons,
                          delta=1e-5, n_runs=5, batch_size=256,
                          n_epochs=30, clip_norm=1.0):
    """
    Sweep epsilon values using DP-SGD and measure accuracy/F1.

    For each epsilon, finds the noise multiplier σ via binary search,
    trains DP-SGD logistic regression, and evaluates on the clean test set.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : numpy arrays
    epsilons  : list of target privacy budgets
    delta     : DP delta parameter
    n_runs    : repetitions per epsilon for confidence intervals
    batch_size: mini-batch size L
    n_epochs  : training epochs
    clip_norm : gradient clipping bound C

    Returns
    -------
    results_list  : list of dicts (one per epsilon)
    baseline_acc  : no-DP accuracy
    """
    n, d = X_train.shape

    # No-DP baseline
    from dp_ml import _LogisticRegression, _accuracy as acc0, _f1 as f10
    X_tr_s, X_te_s = _standardize(X_train, X_test)[:2]
    clf0 = _LogisticRegression()
    clf0.fit(X_tr_s, y_train)
    y_pred0 = clf0.predict(X_te_s)
    base_acc = _accuracy(y_test, y_pred0)
    base_f1 = _f1(y_test, y_pred0)

    results = [{
        'epsilon': float('inf'),
        'actual_epsilon': float('inf'),
        'mechanism': 'dp_sgd',
        'sigma': 0.0,
        'accuracy': base_acc,
        'f1_score': base_f1,
        'accuracy_std': 0.0,
        'f1_std': 0.0,
        'relative_accuracy_loss_pct': 0.0,
        'privacy_type': 'no_dp',
    }]

    # Normalise train data
    X_tr_s, X_te_s, mu, sd = _standardize(X_train, X_test)

    for eps in epsilons:
        sigma, actual_eps = find_sigma_for_epsilon(
            eps, n, batch_size, n_epochs, delta
        )

        accs, f1s = [], []
        for _ in range(n_runs):
            clf = DPSGDLogisticRegression(
                clip_norm=clip_norm,
                sigma=sigma,
                lr=0.05,
                batch_size=batch_size,
                n_epochs=n_epochs,
            )
            clf.fit(X_tr_s, y_train)
            y_pred = clf.predict(X_te_s)
            accs.append(_accuracy(y_test, y_pred))
            f1s.append(_f1(y_test, y_pred))

        mean_acc = float(np.mean(accs))
        results.append({
            'epsilon': float(eps),
            'actual_epsilon': float(actual_eps),
            'mechanism': 'dp_sgd',
            'sigma': float(sigma),
            'accuracy': mean_acc,
            'f1_score': float(np.mean(f1s)),
            'accuracy_std': float(np.std(accs)),
            'f1_std': float(np.std(f1s)),
            'relative_accuracy_loss_pct': (base_acc - mean_acc) / max(base_acc, 1e-8) * 100.0,
            'privacy_type': 'central_dp',
        })

    return results, base_acc
