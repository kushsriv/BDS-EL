"""
Local Differential Privacy (LDP) mechanisms.

In the LOCAL model each user perturbs their own value before sending it
to the aggregator.  No trusted data curator is required, making it more
privacy-preserving than central DP — but at the cost of much higher
noise (O(√n) factor worse in MSE).

Implements:
  1. Laplace LDP          — trivial baseline
  2. Duchi et al. (2013)  — MSE-optimal for real-valued data, low epsilon
  3. Wang et al. (2019)   — piecewise mechanism, beats Duchi for ε > ~0.61
  4. Randomized Response  — Warner (1965) for categorical features

All mechanisms satisfy ε-LDP.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------

def laplace_ldp(value, sensitivity, epsilon):
    """Laplace LDP: user adds Lap(0, sensitivity/ε) locally."""
    return float(value) + np.random.laplace(0, sensitivity / epsilon)


def duchi_mechanism(value, epsilon, data_min=0.0, data_max=1.0):
    """
    Duchi, Jordan & Wainwright (2013) optimal LDP mechanism.

    Normalises input to [-1, 1], samples from {-C, +C} with probability
    proportional to the true value.  Unbiased, ε-LDP, MSE-optimal.

    C = (exp(ε) + 1) / (exp(ε) - 1)
    """
    if data_max <= data_min:
        return float(value)
    v = np.clip(2.0 * (value - data_min) / (data_max - data_min) - 1.0, -1.0, 1.0)
    exp_e = np.exp(float(epsilon))
    C = (exp_e + 1.0) / (exp_e - 1.0)
    p_pos = np.clip((1.0 + v * (exp_e - 1.0) / (exp_e + 1.0)) / 2.0, 0.0, 1.0)
    z = C if np.random.random() < p_pos else -C
    return (z + 1.0) / 2.0 * (data_max - data_min) + data_min


def piecewise_mechanism(value, epsilon, data_min=0.0, data_max=1.0):
    """
    Wang et al. (2019) piecewise mechanism.

    Outputs a value sampled from a piecewise-uniform distribution.
    Satisfies ε-LDP, unbiased, and achieves lower MSE than Duchi when
    ε > 0.61.  Widely used in practice (e.g., Apple's local DP).

    Uses C = (exp(ε/2) + 1) / (exp(ε/2) - 1).
    """
    if data_max <= data_min:
        return float(value)
    v = np.clip(2.0 * (value - data_min) / (data_max - data_min) - 1.0, -1.0, 1.0)
    exp_e2 = np.exp(float(epsilon) / 2.0)
    C = (exp_e2 + 1.0) / (exp_e2 - 1.0)
    l_val = (C + 1.0) / 2.0 * v - (C - 1.0) / 2.0
    r_val = l_val + C - 1.0
    p_mid = exp_e2 / (exp_e2 + 1.0)

    if np.random.random() < p_mid:
        z = np.random.uniform(l_val, r_val)
    else:
        left_w = max(l_val - (-C), 0.0)
        right_w = max(C - r_val, 0.0)
        total = left_w + right_w
        if total < 1e-12:
            z = np.random.choice([-C, C])
        elif np.random.random() < left_w / total:
            z = np.random.uniform(-C, l_val) if l_val > -C else -C
        else:
            z = np.random.uniform(r_val, C) if r_val < C else C

    z = float(np.clip(z, -C, C))
    return (z + 1.0) / 2.0 * (data_max - data_min) + data_min


def randomized_response(value, epsilon, domain_size=2):
    """
    Warner's Randomized Response (1965) extended to k-ary domains.

    Returns true value with probability exp(ε) / (exp(ε) + k - 1),
    otherwise a uniformly random other value.  Satisfies ε-LDP.
    """
    k = max(2, int(domain_size))
    exp_e = np.exp(float(epsilon))
    p_true = exp_e / (exp_e + k - 1)
    p_other = 1.0 / (exp_e + k - 1)
    probs = np.full(k, p_other)
    probs[int(value) % k] = p_true
    probs /= probs.sum()
    return int(np.random.choice(k, p=probs))


# ---------------------------------------------------------------------------
# Theoretical MSE
# ---------------------------------------------------------------------------

def ldp_mse_laplace(epsilon, n, sensitivity=1.0):
    """Var of Laplace-LDP mean estimator: 2(s/ε)²/n."""
    return 2.0 * (sensitivity / epsilon) ** 2 / n


def ldp_mse_duchi(epsilon, n, data_range=1.0):
    """Var of Duchi mean estimator (in original space)."""
    exp_e = np.exp(float(epsilon))
    C = (exp_e + 1.0) / (exp_e - 1.0)
    return (C * data_range / 2.0) ** 2 / n


def ldp_mse_piecewise(epsilon, n, data_range=1.0):
    """Var of piecewise mean estimator."""
    exp_e2 = np.exp(float(epsilon) / 2.0)
    C = (exp_e2 + 1.0) / (exp_e2 - 1.0)
    return (C * data_range / 2.0) ** 2 / n


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_ldp_experiment(data, epsilon, mechanism='laplace',
                        sensitivity=1.0, n_runs=30):
    """
    Apply an LDP mechanism to all records and measure mean-estimation error.

    Returns dict with mean_error, std_error, theoretical_mse, and
    the MSE ratio vs. central DP (to quantify the LDP penalty).
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return {}

    true_mean = float(np.mean(data))
    n = len(data)
    lo, hi = float(np.min(data)), float(np.max(data))
    r = hi - lo

    errors = []
    for _ in range(n_runs):
        if mechanism == 'laplace':
            noisy = np.array([laplace_ldp(v, sensitivity, epsilon) for v in data])
        elif mechanism == 'duchi':
            noisy = np.array([duchi_mechanism(v, epsilon, lo, hi) for v in data])
        elif mechanism == 'piecewise':
            noisy = np.array([piecewise_mechanism(v, epsilon, lo, hi) for v in data])
        else:
            noisy = np.array([laplace_ldp(v, sensitivity, epsilon) for v in data])
        errors.append(abs(float(np.mean(noisy)) - true_mean))

    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))

    # Theoretical MSE for LDP vs central-DP Laplace
    if mechanism == 'laplace':
        theo_mse = ldp_mse_laplace(epsilon, n, sensitivity)
    elif mechanism == 'duchi':
        theo_mse = ldp_mse_duchi(epsilon, n, r)
    elif mechanism == 'piecewise':
        theo_mse = ldp_mse_piecewise(epsilon, n, r)
    else:
        theo_mse = ldp_mse_laplace(epsilon, n, sensitivity)

    central_mse = 2.0 * (sensitivity / epsilon) ** 2  # central DP (no /n factor)
    # Central DP adds noise once to the aggregate; LDP adds it n times
    # For a fair comparison the central MSE for the mean is 2*(s/eps)^2 / n^2
    # (noise on sum divided by n), while LDP is 2*(s/eps)^2 / n.
    # Ratio = n (the well-known sqrt(n) worse error in std).
    central_dp_mse_mean = 2.0 * (sensitivity / epsilon) ** 2 / (n ** 2)
    ldp_penalty = theo_mse / central_dp_mse_mean if central_dp_mse_mean > 0 else float('nan')

    return {
        'epsilon': float(epsilon),
        'mechanism': mechanism,
        'privacy_model': 'local',
        'mean_error': mean_err,
        'std_error': std_err,
        'theoretical_mse': float(np.sqrt(theo_mse)),  # as RMSE for direct comparison
        'ldp_penalty_vs_central': float(ldp_penalty),
    }
