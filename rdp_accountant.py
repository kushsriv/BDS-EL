"""
Rényi Differential Privacy (RDP) accountant.

Tracks cumulative privacy budget across multiple queries and provides
tighter (ε, δ)-DP bounds than basic composition.

Key references:
  Mironov (2017)  "Renyi Differential Privacy of the Gaussian Mechanism"
  Balle et al. (2020) "Hypothesis Testing Interpretations and Renyi DP"

RDP composes additively: after k queries the total RDP at order α
equals the sum of per-query RDP values at α.  Converting via

    ε(δ) = RDP(α) + log(1/δ) / (α - 1)

and minimising over α gives a bound much tighter than ε_basic = Σ εᵢ
when many queries are composed.
"""
import numpy as np

STANDARD_ALPHAS = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 16.0, 32.0, 64.0]


# ---------------------------------------------------------------------------
# Per-mechanism RDP
# ---------------------------------------------------------------------------

def rdp_laplace(alpha, epsilon):
    """
    RDP of the Laplace mechanism at order α.

    For Lap(sensitivity/ε) with sensitivity=1 (Mironov 2017, Prop. 3):

        ε_RDP(α) = (1/(α-1)) log(
            α/(2α-1) · exp((α-1)ε)  +  (α-1)/(2α-1) · exp(-αε)
        )

    Limits: ε_RDP(1) = ε_RDP(∞) = ε  (matches pure DP parameter).
    """
    if alpha == 1 or np.isinf(alpha):
        return float(epsilon)
    a, e = float(alpha), float(epsilon)
    # Guard overflow for large alpha*epsilon (dominant term → e + const/a)
    if (a - 1) * e > 500:
        return float(e + np.log(a / (2 * a - 1)) / (a - 1))
    term1 = a / (2 * a - 1) * np.exp((a - 1) * e)
    term2 = (a - 1) / (2 * a - 1) * np.exp(-a * e)
    log_val = np.log(max(term1 + term2, 1e-300))
    return float(log_val / (a - 1))


def rdp_gaussian(alpha, sigma, sensitivity=1.0):
    """
    RDP of the Gaussian mechanism at order α (Mironov 2017, Prop. 3).

        ε_RDP(α) = α · sensitivity² / (2 σ²)
    """
    return float(alpha * (sensitivity ** 2) / (2.0 * sigma ** 2))


def rdp_subsampled_laplace(alpha, epsilon, q):
    """
    RDP of Poisson-subsampled Laplace mechanism (amplification by sampling).

    Sampling fraction q gives:  ε_amp = log(1 + q(exp(ε) - 1))
    then  ε_RDP(α) computed for the amplified mechanism.
    """
    if q >= 1.0:
        return rdp_laplace(alpha, epsilon)
    if q <= 0.0:
        return 0.0
    eps_amp = np.log(1.0 + q * (np.exp(float(epsilon)) - 1.0))
    return rdp_laplace(alpha, eps_amp)


# ---------------------------------------------------------------------------
# RDP → (ε, δ)-DP conversion
# ---------------------------------------------------------------------------

def rdp_to_dp(alpha, rdp_eps, delta):
    """
    Convert RDP guarantee (α, ε_RDP) to (ε, δ)-DP.

        ε(δ) = ε_RDP + log(1/δ) / (α - 1)   [Mironov 2017, Prop. 3]
    """
    if alpha <= 1:
        return float('inf')
    if np.isinf(alpha):
        return float(rdp_eps)
    return float(rdp_eps + np.log(1.0 / delta) / (alpha - 1))


def best_dp_from_rdp_curve(rdp_curve, delta):
    """
    Given {α: ε_RDP} pairs, find the tightest (ε, δ)-DP by optimising over α.

    Returns (best_epsilon, best_alpha).
    """
    best_eps = float('inf')
    best_alpha = None
    for a, rdp_eps in rdp_curve.items():
        if a <= 1 or np.isinf(rdp_eps):
            continue
        eps = rdp_to_dp(a, rdp_eps, delta)
        if eps < best_eps:
            best_eps = eps
            best_alpha = a
    return best_eps, best_alpha


# ---------------------------------------------------------------------------
# Advanced composition theorem (for comparison)
# ---------------------------------------------------------------------------

def advanced_composition_bound(n_queries, epsilon, delta, delta_prime=1e-6):
    """
    Advanced composition theorem (Dwork, Rothblum, Vadhan 2010).

    After k queries each with (ε, δ')-DP, the composition satisfies
    (ε', k·δ' + δ)-DP where:

        ε' = sqrt(2k · log(1/δ)) · ε  +  k · ε · (exp(ε) - 1)

    More practical simplified form used here:
        ε' ≈ epsilon * sqrt(2 * k * log(1/delta))
    """
    k = n_queries
    eps_adv = epsilon * np.sqrt(2 * k * np.log(1.0 / delta)) + k * epsilon * (np.exp(epsilon) - 1)
    return float(eps_adv), float(k * delta_prime + delta)


# ---------------------------------------------------------------------------
# Accountant class
# ---------------------------------------------------------------------------

class RDPAccountant:
    """
    Cumulative RDP tracker for a sequence of DP queries.

    Usage:
        acc = RDPAccountant()
        acc.add_laplace_query(epsilon=1.0, sensitivity=GS)
        acc.add_laplace_query(epsilon=0.5, sensitivity=GS)
        eps, alpha = acc.get_privacy_spent(delta=1e-5)
    """

    def __init__(self, alphas=None):
        self.alphas = list(alphas or STANDARD_ALPHAS)
        self._rdp = {a: 0.0 for a in self.alphas}
        self.queries = []

    def add_laplace_query(self, epsilon, sensitivity=1.0, sampling_rate=1.0):
        """Record a Laplace mechanism query (optionally with subsampling)."""
        eff_eps = float(epsilon) / float(sensitivity) if sensitivity != 1.0 else float(epsilon)
        for a in self.alphas:
            if sampling_rate < 1.0:
                self._rdp[a] += rdp_subsampled_laplace(a, eff_eps, sampling_rate)
            else:
                self._rdp[a] += rdp_laplace(a, eff_eps)
        self.queries.append({
            'type': 'laplace', 'epsilon': epsilon,
            'sensitivity': sensitivity, 'sampling_rate': sampling_rate,
        })

    def add_gaussian_query(self, epsilon, sensitivity=1.0, delta=1e-5):
        """Record a Gaussian mechanism query."""
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * float(sensitivity) / float(epsilon)
        for a in self.alphas:
            self._rdp[a] += rdp_gaussian(a, sigma, sensitivity)
        self.queries.append({
            'type': 'gaussian', 'epsilon': epsilon,
            'sensitivity': sensitivity, 'sigma': sigma,
        })

    def get_privacy_spent(self, delta=1e-5):
        """Return total (ε, δ)-DP via RDP composition (minimised over α)."""
        return best_dp_from_rdp_curve(self._rdp, delta)

    def get_basic_composition(self):
        """Naive basic composition bound: Σ εᵢ."""
        return sum(float(q.get('epsilon') or 0) for q in self.queries)

    def get_rdp_curve(self):
        return dict(self._rdp)

    def composition_comparison(self, delta=1e-5):
        """Dict comparing basic, advanced, and RDP composition bounds."""
        k = len(self.queries)
        if k == 0:
            return {}
        epsilons = [float(q.get('epsilon') or 0) for q in self.queries]
        avg_eps = float(np.mean(epsilons)) if epsilons else 0.0

        basic_eps = self.get_basic_composition()
        rdp_eps, best_alpha = self.get_privacy_spent(delta)
        adv_eps, _ = advanced_composition_bound(k, avg_eps, delta)

        return {
            'n_queries': k,
            'basic_composition_epsilon': basic_eps,
            'advanced_composition_epsilon': adv_eps,
            'rdp_composition_epsilon': rdp_eps,
            'best_rdp_alpha': best_alpha,
            'rdp_vs_basic_tightening': basic_eps / rdp_eps if rdp_eps > 0 else float('inf'),
            'adv_vs_basic_tightening': basic_eps / adv_eps if adv_eps > 0 else float('inf'),
            'delta': delta,
        }

    def reset(self):
        self._rdp = {a: 0.0 for a in self.alphas}
        self.queries = []
