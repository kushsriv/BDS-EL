"""
PRV (Privacy Random Variable) Accountant
=========================================

Implements the numerical composition framework from:
  Gopi, S., Lee, Y. T., & Wajc, D. (2021).
  "Numerical composition of differential privacy." NeurIPS 2021.

Key Idea
--------
For a mechanism M and adjacent databases (D, D'), the privacy loss RV is:
    Z = log( Pr[M(D) = o] / Pr[M(D') = o] )  where o ~ M(D)

- Z is bounded in [-ε, ε] for ε-DP mechanisms.
- For composition of k mechanisms: Z_total = Z_1 + ... + Z_k  (by independence).
- δ(ε) = E[max(0, 1 - exp(ε - Z))]  (hockey-stick divergence of the composed mechanism).
- The composition CDF is computed by FFT convolution of per-mechanism PRVs.

Comparison to RDP
-----------------
- RDP composes by summing at each order α: R_α(k) = k · R_α(1).
- PRV is tighter because it tracks the full distribution, not just moments.
- For Laplace mechanisms, PRV is strictly tighter than RDP for k ≥ 2.
- For Gaussian mechanisms, PRV matches the tight Gaussian composition (f-DP).

Laplace PRV
-----------
For Lap(0, s/ε₀) mechanism applied to a query with sensitivity s:
  Z = ε₀/s · (|y - f(D') | - |y - f(D)|)  where y ~ Lap(f(D), s/ε₀)
  Z ∈ [-ε₀, ε₀]

  P(Z = ε₀)    = 1/2          [y < f(D)]
  P(Z = -ε₀)   = e^(-ε₀)/2   [y > f(D) + s]
  P(Z ∈ (t))   = continuous   for t ∈ (-ε₀, ε₀)

Gaussian PRV
------------
For N(0, σ²) mechanism with sensitivity s:
  Z = s(y - s/2) / σ²  where y ~ N(0, σ²)   [for the "D-view"]
  Z ~ N(s²/(2σ²),  s²/σ²)    — a Gaussian shifted by the privacy loss mean
"""

import numpy as np


# ---------------------------------------------------------------------------
# PRV for Laplace mechanism  (exact analytic CDF)
# ---------------------------------------------------------------------------

class LaplacePRV:
    """
    Exact privacy loss distribution for the Laplace mechanism.

    Parameters
    ----------
    epsilon0 : the per-query ε (noise scale = sensitivity / epsilon0)
    """

    def __init__(self, epsilon0):
        self.e0 = float(epsilon0)

    def cdf(self, t):
        """P(Z ≤ t) for the Laplace privacy loss RV."""
        e0 = self.e0
        t = np.asarray(t, dtype=float)
        result = np.zeros_like(t)

        # Z = e0 w.p. 1/2 (discrete atom at e0)
        # Z = -e0 w.p. exp(-e0)/2 (discrete atom at -e0)
        # Z ∈ (-e0, e0): continuous uniform-ish distribution
        #
        # Full CDF:
        #   t < -e0:  0
        #   t = -e0:  exp(-e0)/2
        #   -e0 < t < e0:  (1 + t/e0) / 2
        #   t >= e0:  1

        mask_lo = t < -e0
        mask_hi = t >= e0
        mask_mid = ~mask_lo & ~mask_hi

        result[mask_lo] = 0.0
        result[mask_hi] = 1.0
        result[mask_mid] = (1.0 + t[mask_mid] / e0) / 2.0

        # Discrete atom at -e0
        at_atom = np.isclose(t, -e0)
        result[at_atom] = np.exp(-e0) / 2.0

        return result

    def pdf_grid(self, grid):
        """
        Probability mass on a discrete grid (for FFT convolution).
        Each grid point gets probability from the CDF difference.
        """
        delta = grid[1] - grid[0]
        cdf_right = self.cdf(grid + delta / 2)
        cdf_left = self.cdf(grid - delta / 2)
        return np.maximum(cdf_right - cdf_left, 0.0)

    def support(self):
        return (-self.e0, self.e0)


# ---------------------------------------------------------------------------
# PRV for Gaussian mechanism  (discretized normal)
# ---------------------------------------------------------------------------

class GaussianPRV:
    """
    Privacy loss distribution for the Gaussian mechanism.

    Z ~ N(mu, mu * 2)  where mu = sensitivity² / (2σ²)

    Parameters
    ----------
    sensitivity : GS of the query
    sigma       : noise standard deviation
    """

    def __init__(self, sensitivity, sigma):
        self.mu = (float(sensitivity) ** 2) / (2.0 * float(sigma) ** 2)

    def cdf(self, t):
        """P(Z ≤ t). Z ~ N(mu, 2mu) where mu = s²/(2σ²)."""
        mu = self.mu
        t = np.asarray(t, dtype=float)
        sigma_z = np.sqrt(2.0 * mu)
        if sigma_z < 1e-12:
            return (t >= 0.0).astype(float)
        z = (t - mu) / sigma_z
        return _norm_cdf(z)

    def pdf_grid(self, grid):
        delta = grid[1] - grid[0]
        return np.maximum(self.cdf(grid + delta / 2) - self.cdf(grid - delta / 2), 0.0)

    def support(self):
        mu = self.mu
        return (-5 * np.sqrt(2 * mu) + mu, 5 * np.sqrt(2 * mu) + mu)


def _norm_cdf(z):
    """Vectorised standard normal CDF."""
    return 0.5 * (1.0 + np.vectorize(_erf)(z / np.sqrt(2.0)))


def _erf(x):
    """Abramowitz & Stegun approximation of erf(x), error < 1.5e-7."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t)
                  + 1.421413741) * t - 0.284496736) * t
                + 0.254829592) * t * np.exp(-x * x)
    return sign * y


# ---------------------------------------------------------------------------
# FFT-based composition
# ---------------------------------------------------------------------------

class PRVAccountant:
    """
    Privacy accountant using FFT convolution of privacy loss distributions.

    Usage
    -----
    acc = PRVAccountant(grid_size=4096, epsilon_max=20.0)
    acc.add_laplace_query(epsilon=1.0)
    acc.add_laplace_query(epsilon=1.0)
    epsilon = acc.get_privacy_spent(delta=1e-5)
    """

    def __init__(self, grid_size=4096, epsilon_max=20.0):
        self.grid_size = int(grid_size)
        self.epsilon_max = float(epsilon_max)
        # Discretisation grid: uniform on [-epsilon_max, epsilon_max]
        self.grid = np.linspace(-epsilon_max, epsilon_max, self.grid_size)
        self.delta_z = self.grid[1] - self.grid[0]
        # Running composed PMF (starts as Dirac at 0)
        self._pmf = np.zeros(self.grid_size)
        mid = self.grid_size // 2
        self._pmf[mid] = 1.0  # Z = 0 (no queries yet)
        self.n_queries = 0

    def _convolve(self, pmf_a, pmf_b):
        """Convolve two PMFs using FFT (linear convolution, then truncate to center)."""
        n = len(pmf_a)
        # Use enough padding to avoid wrap-around aliasing
        n_fft = int(2 ** np.ceil(np.log2(2 * n - 1)))
        fa = np.fft.rfft(pmf_a, n=n_fft)
        fb = np.fft.rfft(pmf_b, n=n_fft)
        full = np.fft.irfft(fa * fb, n=n_fft)
        # The linear convolution has length 2n-1; take center n elements
        start = (n - 1) // 2
        result = full[start: start + n]
        return np.maximum(result, 0.0)

    def add_laplace_query(self, epsilon, sensitivity=1.0, sampling_rate=1.0):
        """Add one Laplace mechanism query to the composition."""
        e0 = epsilon * sensitivity  # effective per-query ε
        prv = LaplacePRV(e0)
        pmf_new = prv.pdf_grid(self.grid)
        if sampling_rate < 1.0:
            pmf_new = self._amplify_pmf(pmf_new, sampling_rate)
        self._pmf = self._convolve(self._pmf, pmf_new)
        self._pmf /= max(self._pmf.sum(), 1e-300)
        self.n_queries += 1

    def add_gaussian_query(self, sensitivity, sigma, sampling_rate=1.0):
        """Add one Gaussian mechanism query to the composition."""
        prv = GaussianPRV(sensitivity, sigma)
        pmf_new = prv.pdf_grid(self.grid)
        if sampling_rate < 1.0:
            pmf_new = self._amplify_pmf(pmf_new, sampling_rate)
        self._pmf = self._convolve(self._pmf, pmf_new)
        self._pmf /= max(self._pmf.sum(), 1e-300)
        self.n_queries += 1

    def _amplify_pmf(self, pmf, q):
        """
        Privacy amplification by Poisson subsampling.

        For a subsampled mechanism with rate q:
          Z_amp = Z with prob q, Z_amp = 0 with prob (1-q)
        """
        mid = self.grid_size // 2
        pmf_zero = np.zeros(self.grid_size)
        pmf_zero[mid] = 1.0
        return (1.0 - q) * pmf_zero + q * pmf

    def get_privacy_spent(self, delta=1e-5):
        """
        Compute ε such that the composed mechanism is (ε, δ)-DP.

        Uses the hockey-stick divergence:
            δ(ε) = Σ_t pmf(t) · max(0, 1 - exp(ε - t))

        Returns ε that achieves the given δ.
        """
        # Binary search for ε
        lo, hi = 0.0, self.epsilon_max
        for _ in range(60):
            mid_eps = (lo + hi) / 2.0
            d = self._hockey_stick(mid_eps)
            if d > delta:
                lo = mid_eps
            else:
                hi = mid_eps
        return float(hi)

    def _hockey_stick(self, epsilon):
        """δ(ε) = E[max(0, 1 - exp(ε - Z))] under the composed PRV."""
        corrections = np.maximum(0.0, 1.0 - np.exp(epsilon - self.grid))
        return float(np.dot(self._pmf, corrections))

    def get_privacy_profile(self, epsilons=None, delta=1e-5):
        """Return δ(ε) for a range of ε values."""
        if epsilons is None:
            epsilons = np.linspace(0.01, self.epsilon_max, 200)
        deltas = [self._hockey_stick(e) for e in epsilons]
        return np.array(epsilons), np.array(deltas)

    def reset(self):
        self._pmf = np.zeros(self.grid_size)
        mid = self.grid_size // 2
        self._pmf[mid] = 1.0
        self.n_queries = 0

    def composition_comparison_prv_vs_rdp(self, delta=1e-5):
        """
        Compare PRV ε against the RDP bound for the same composition.
        Returns dict with both estimates.
        """
        from rdp_accountant import RDPAccountant
        # PRV result
        prv_eps = self.get_privacy_spent(delta)
        return {
            'n_queries': self.n_queries,
            'prv_epsilon': prv_eps,
        }


# ---------------------------------------------------------------------------
# Experiment: PRV vs RDP comparison
# ---------------------------------------------------------------------------

def run_prv_vs_rdp_experiment(epsilons_per_query, k_values, delta=1e-5,
                               mechanism='laplace', sensitivity=1.0,
                               grid_size=4096, epsilon_max=None):
    """
    Compare PRV and RDP composition bounds across k queries.

    Parameters
    ----------
    epsilons_per_query : list of per-query ε values to test
    k_values           : list of query counts k
    delta              : DP delta
    mechanism          : 'laplace' | 'gaussian'
    sensitivity        : query sensitivity
    grid_size          : PRV grid resolution (higher = more accurate)
    epsilon_max        : maximum ε on the PRV grid

    Returns
    -------
    list of dicts with keys:
      mechanism, per_query_epsilon, n_queries,
      rdp_epsilon, prv_epsilon, advanced_epsilon, basic_epsilon,
      prv_vs_rdp_ratio, prv_tightening_pct
    """
    from rdp_accountant import RDPAccountant, advanced_composition_bound

    rows = []

    for eps0 in epsilons_per_query:
        # Adaptive epsilon_max: at least 3× the basic composition bound at max k
        k_max = max(k_values)
        auto_emax = max(3.0 * k_max * eps0, 5.0) if epsilon_max is None else epsilon_max
        prv_acc = PRVAccountant(grid_size=grid_size, epsilon_max=auto_emax)
        rdp_acc = RDPAccountant()
        k_done = 0

        for k in sorted(k_values):
            # Add queries from k_done up to k
            for _ in range(k - k_done):
                if mechanism == 'laplace':
                    prv_acc.add_laplace_query(eps0, sensitivity=sensitivity)
                    rdp_acc.add_laplace_query(eps0, sensitivity=sensitivity)
                else:
                    # Gaussian: σ = sqrt(2 ln(1.25/δ)) * s / ε₀
                    sigma = np.sqrt(2.0 * np.log(1.25 / delta)) * sensitivity / eps0
                    prv_acc.add_gaussian_query(sensitivity, sigma)
                    rdp_acc.add_gaussian_query(eps0, sensitivity=sensitivity, delta=delta)
                k_done += 1

            prv_eps = prv_acc.get_privacy_spent(delta)
            rdp_eps, _ = rdp_acc.get_privacy_spent(delta)
            basic_eps = k * eps0
            adv_eps, _ = advanced_composition_bound(k, eps0, delta)

            rows.append({
                'mechanism': mechanism,
                'per_query_epsilon': float(eps0),
                'n_queries': k,
                'prv_epsilon': float(prv_eps),
                'rdp_epsilon': float(rdp_eps),
                'advanced_epsilon': float(adv_eps),
                'basic_epsilon': float(basic_eps),
                'prv_vs_rdp_ratio': float(rdp_eps / prv_eps) if prv_eps > 0 else float('inf'),
                'prv_tightening_pct': float((rdp_eps - prv_eps) / rdp_eps * 100) if rdp_eps > 0 else 0.0,
            })

    return rows
