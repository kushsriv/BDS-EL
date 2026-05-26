"""
Core DP noise utilities.

Changes from v1:
  - Data-driven sensitivity via `add_dp_with_data_sensitivity`
  - Privacy amplification by subsampling (Poisson model)
  - Noise variance helper for theoretical analysis
"""
import numpy as np
from sensitivity import compute_global_sensitivity


# ---------------------------------------------------------------------------
# Primitive noise generators
# ---------------------------------------------------------------------------

def laplace_noise(scale):
    return np.random.laplace(0, scale)


def gaussian_noise(sigma):
    return np.random.normal(0, sigma)


def exponential_mechanism(scores, epsilon, sensitivity=1.0):
    """DP selection from a scored candidate set."""
    utilities = np.array([s[0] for s in scores])
    values = [s[1] for s in scores]
    exp_scores = np.exp(epsilon * utilities / (2.0 * sensitivity))
    probs = exp_scores / exp_scores.sum()
    return values[np.random.choice(len(values), p=probs)]


# ---------------------------------------------------------------------------
# Add DP noise to a scalar value
# ---------------------------------------------------------------------------

def add_dp(value, sensitivity, epsilon, mechanism='laplace', delta=1e-5):
    """
    Add DP noise to `value`.

    Laplace: noise ~ Lap(0, sensitivity/ε)            → pure ε-DP
    Gaussian: noise ~ N(0, σ²) with σ = √(2ln(1.25/δ)) · s/ε  → (ε,δ)-DP
    """
    if mechanism == 'laplace':
        return float(value) + laplace_noise(sensitivity / epsilon)
    elif mechanism == 'gaussian':
        sigma = np.sqrt(2.0 * np.log(1.25 / delta)) * sensitivity / epsilon
        return float(value) + gaussian_noise(sigma)
    else:
        raise ValueError(f'Unknown mechanism: {mechanism}')


def add_dp_with_data_sensitivity(value, data, epsilon,
                                  mechanism='laplace', delta=1e-5, query='mean'):
    """
    Like `add_dp` but calibrates sensitivity from the actual data distribution.

    This replaces the fixed sensitivity=1.0 assumption with the correct
    global sensitivity for the query on the given dataset.
    """
    sensitivity = compute_global_sensitivity(data, query)
    return add_dp(value, sensitivity, epsilon, mechanism, delta)


def add_noise_to_record(value, sensitivity, epsilon, mechanism='laplace', delta=1e-5):
    return add_dp(value, sensitivity, epsilon, mechanism, delta)


# ---------------------------------------------------------------------------
# Privacy amplification by subsampling
# ---------------------------------------------------------------------------

def privacy_amplification_subsampling(epsilon, delta, sampling_rate):
    """
    Poisson subsampling amplification (Balle et al. 2018).

    Sampling each record with probability q before applying (ε, δ)-DP gives
    effective:
        ε_amp  = log(1 + q(exp(ε) - 1))   ≈ q·ε  for small ε
        δ_amp  = q · δ

    Returns (epsilon_amplified, delta_amplified).
    """
    q = float(sampling_rate)
    if q <= 0:
        return 0.0, 0.0
    if q >= 1:
        return float(epsilon), float(delta)
    eps_amp = np.log(1.0 + q * (np.exp(float(epsilon)) - 1.0))
    return float(eps_amp), float(q * delta)


# ---------------------------------------------------------------------------
# Noise variance (for theoretical analysis)
# ---------------------------------------------------------------------------

def compute_noise_variance(sensitivity, epsilon, mechanism='laplace', delta=1e-5):
    """
    Return the variance of the noise added by `mechanism`.

    Laplace Lap(0, b): Var = 2b²
    Gaussian N(0, σ²): Var = σ²
    """
    if mechanism == 'laplace':
        b = sensitivity / epsilon
        return 2.0 * b ** 2
    elif mechanism == 'gaussian':
        sigma = np.sqrt(2.0 * np.log(1.25 / delta)) * sensitivity / epsilon
        return sigma ** 2
    return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def repeat_runs(func, n_runs, *args, **kwargs):
    """Call `func` n_runs times and return results as a list."""
    return [func(*args, **kwargs) for _ in range(n_runs)]
