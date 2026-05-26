"""
Membership Inference Attack (MIA) simulation for DP validation.

Purpose: empirically validate DP guarantees beyond theoretical claims.
A good DP mechanism should limit the attacker's advantage over random
guessing to near zero, especially at small epsilon.

Attack model (Yeom et al. 2018):
  The attacker observes a DP aggregate M(D) and must decide whether a
  target record t is in D (IN) or in D' = D \ {t} ∪ {t'} (OUT).

Optimal attacker: likelihood-ratio test (Neyman-Pearson optimal).
  Predict IN  iff  log Pr[M(D)=o | IN] - log Pr[M(D')=o | OUT] > 0

DP guarantee implies:
  Pr[correct] ≤ exp(ε) / (1 + exp(ε))
  Advantage    ≤ (exp(ε) - 1) / (2(exp(ε) + 1))

By plotting empirical advantage alongside the theoretical bound we can
verify that the implementation is correct and show how privacy degrades
as epsilon grows.
"""
import numpy as np


# ---------------------------------------------------------------------------
# PDF helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _laplace_log_pdf(x, mu, scale):
    return -np.log(2.0 * scale) - np.abs(x - mu) / scale


def _gaussian_log_pdf(x, mu, sigma):
    return -0.5 * np.log(2.0 * np.pi * sigma ** 2) - (x - mu) ** 2 / (2.0 * sigma ** 2)


# ---------------------------------------------------------------------------
# Theoretical bounds
# ---------------------------------------------------------------------------

def theoretical_mia_bound(epsilon):
    """
    Yeom et al. (2018) upper bound on MIA success from ε-DP.

    Returns (max_accuracy, max_advantage).
    """
    exp_e = np.exp(float(epsilon))
    max_acc = exp_e / (1.0 + exp_e)
    return float(max_acc), float(max_acc - 0.5)


# ---------------------------------------------------------------------------
# Core attack
# ---------------------------------------------------------------------------

def run_mia_experiment(true_values, epsilon, sensitivity=1.0,
                        mechanism='laplace', delta=1e-5, n_trials=1000):
    """
    Simulate a likelihood-ratio MIA against a single DP query.

    Scenario
    --------
    Dataset D has `true_values`.  The DP mechanism outputs:
      IN  sample:  M(D)  = mean(D) + noise
      OUT sample:  M(D') = mean(D') + noise  where D' swaps one record
                   by adding `sensitivity` to it.

    The attacker observes one sample, computes the LLR, and guesses IN or OUT.

    Returns
    -------
    dict with empirical accuracy, advantage, AUC, and theoretical bounds.
    """
    data = np.asarray(true_values, dtype=float)
    n = len(data)
    if n == 0:
        return {}

    mu_in = float(np.mean(data))
    # Adjacent dataset: replace first record with record + sensitivity
    data_out = data.copy()
    data_out[0] += sensitivity
    mu_out = float(np.mean(data_out))

    if mechanism == 'laplace':
        scale = sensitivity / epsilon

        def sample_in():
            return mu_in + np.random.laplace(0, scale)

        def sample_out():
            return mu_out + np.random.laplace(0, scale)

        def llr(obs):
            return _laplace_log_pdf(obs, mu_in, scale) - _laplace_log_pdf(obs, mu_out, scale)
    else:
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon

        def sample_in():
            return mu_in + np.random.normal(0, sigma)

        def sample_out():
            return mu_out + np.random.normal(0, sigma)

        def llr(obs):
            return _gaussian_log_pdf(obs, mu_in, sigma) - _gaussian_log_pdf(obs, mu_out, sigma)

    llr_scores = np.empty(2 * n_trials)
    labels = np.empty(2 * n_trials, dtype=int)

    for i in range(n_trials):
        obs_in = sample_in()
        llr_scores[2 * i] = llr(obs_in)
        labels[2 * i] = 1  # member

        obs_out = sample_out()
        llr_scores[2 * i + 1] = llr(obs_out)
        labels[2 * i + 1] = 0  # non-member

    predictions = (llr_scores > 0).astype(int)
    emp_acc = float(np.mean(predictions == labels))
    emp_adv = emp_acc - 0.5

    # Trapezoid AUC (no sklearn)
    order = np.argsort(-llr_scores)
    sorted_labels = labels[order]
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    tpr = np.cumsum(sorted_labels == 1) / max(n_pos, 1)
    fpr = np.cumsum(sorted_labels == 0) / max(n_neg, 1)
    auc = float(np.clip(np.trapezoid(tpr, fpr) if hasattr(np, 'trapezoid') else np.trapz(tpr, fpr), 0.0, 1.0))

    theo_acc, theo_adv = theoretical_mia_bound(epsilon)

    return {
        'epsilon': float(epsilon),
        'mechanism': mechanism,
        'empirical_accuracy': emp_acc,
        'empirical_advantage': emp_adv,
        'empirical_auc': auc,
        'theoretical_max_accuracy': theo_acc,
        'theoretical_max_advantage': theo_adv,
        'privacy_margin': theo_acc - emp_acc,  # positive = mechanism beats bound
    }


def sweep_mia(true_values, epsilons, mechanisms, sensitivity=1.0,
              delta=1e-5, n_trials=1000):
    """Run MIA for every (ε, mechanism) pair."""
    results = []
    for mech in mechanisms:
        for eps in epsilons:
            r = run_mia_experiment(
                true_values, eps,
                sensitivity=sensitivity, mechanism=mech,
                delta=delta, n_trials=n_trials,
            )
            if r:
                results.append(r)
    return results
