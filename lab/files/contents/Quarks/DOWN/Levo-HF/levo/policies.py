import math
from typing import Optional

import numpy as np


def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    x = x / tau
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    if s <= 0.0:
        # fallback: uniform
        return np.ones_like(ex) / len(ex)
    return ex / s


def hf_levo_policy_tabular(
    Q: np.ndarray,
    state: int,
    t: int,
    A: float = 0.5,
    omega: float = 0.1,
    phase_offset: float = 0.0,
    ent_weight: float = 0.0,
    prior: Optional[np.ndarray] = None,
    tau: float = 1.0,
) -> np.ndarray:
    """
    HF-Levo policy on a single Q-table.

    score(a) = Q(s, a) + A cos(omega * t + phi_a + phase_offset)
    """
    q_s = np.asarray(Q[state], dtype=float)
    n_actions = q_s.shape[0]

    phases = np.linspace(0.0, 2.0 * math.pi, n_actions, endpoint=False)
    osc = A * np.cos(omega * float(t) + phases + phase_offset)
    score = q_s + osc

    p = softmax(score, tau=tau)

    if prior is None or ent_weight <= 0.0:
        return p

    prior = np.asarray(prior, dtype=float)
    prior = prior / max(prior.sum(), 1e-8)
    w = max(0.0, min(1.0, float(ent_weight)))
    p_mix = (1.0 - w) * p + w * prior
    p_mix = np.clip(p_mix, 1e-8, 1.0)
    p_mix /= p_mix.sum()
    return p_mix


def ensemble_levo_thinking_policy(
    Q_ensemble: np.ndarray,
    state: int,
    t: int,
    A: float = 0.5,
    omega: float = 0.1,
    phase_offset: float = 0.0,
    lambda_var: float = 0.5,
    prior: Optional[np.ndarray] = None,
    tau: float = 1.0,
) -> np.ndarray:
    """
    LevoThinking ensemble policy.

    Let Q_h be the Q-table of head h.

    mu(a)  = mean_h Q_h(s, a)
    var(a) = var_h  Q_h(s, a)

    base_score(a) = mu(a) - lambda_var * var(a)
    score(a)      = base_score(a) + A cos(omega * t + phi_a + phase_offset)
    """
    q_heads = np.asarray(Q_ensemble[:, state, :], dtype=float)
    mu = q_heads.mean(axis=0)
    var = q_heads.var(axis=0)

    n_actions = mu.shape[0]
    phases = np.linspace(0.0, 2.0 * math.pi, n_actions, endpoint=False)
    osc = A * np.cos(omega * float(t) + phases + phase_offset)

    base_score = mu - lambda_var * var
    score = base_score + osc

    p = softmax(score, tau=tau)

    if prior is None:
        return p

    prior = np.asarray(prior, dtype=float)
    prior = prior / max(prior.sum(), 1e-8)
    p_mix = 0.8 * p + 0.2 * prior
    p_mix = np.clip(p_mix, 1e-8, 1.0)
    p_mix /= p_mix.sum()
    return p_mix
