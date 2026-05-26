import numpy as np

def laplace_noise(scale):
    """Generate Laplace noise with 0 mean and given scale."""
    return np.random.laplace(0, scale)

def gaussian_noise(sigma):
    """Generate Gaussian noise with 0 mean and given sigma."""
    return np.random.normal(0, sigma)

def exponential_mechanism(scores, epsilon):
    """Exponential mechanism for DP selection (for demonstration)."""
    # scores: list of (utility, value) tuples
    utilities = np.array([s[0] for s in scores])
    values = [s[1] for s in scores]
    exp_scores = np.exp(epsilon * utilities / 2)
    probs = exp_scores / np.sum(exp_scores)
    idx = np.random.choice(len(values), p=probs)
    return values[idx]

def add_dp(value, sensitivity, epsilon, mechanism="laplace", delta=1e-5):
    """Add DP noise to a value using the specified mechanism."""
    if mechanism == "laplace":
        scale = sensitivity / epsilon
        noise = laplace_noise(scale)
        return value + noise
    elif mechanism == "gaussian":
        # Gaussian mechanism: sigma = sqrt(2*log(1.25/delta)) * sensitivity / epsilon
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        noise = gaussian_noise(sigma)
        return value + noise
    else:
        raise ValueError(f"Unknown DP mechanism: {mechanism}")

def add_noise_to_record(value, sensitivity, epsilon, mechanism="laplace", delta=1e-5):
    return add_dp(value, sensitivity, epsilon, mechanism, delta)

def repeat_runs(func, n_runs, *args, **kwargs):
    """Repeat a function n_runs times, return list of results."""
    results = []
    for _ in range(n_runs):
        results.append(func(*args, **kwargs))
    return results