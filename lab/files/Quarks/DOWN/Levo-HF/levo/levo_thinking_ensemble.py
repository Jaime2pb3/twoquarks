from typing import Iterable, Optional

import numpy as np

from .policies import ensemble_levo_thinking_policy


class LevoThinkingEnsembleAgent:
    """
    Ensemble Q-learning agent with LevoThinking policy.

    Q has shape (n_heads, n_states, n_actions).

    For action selection, we use:
        mu(a)  = mean_h Q_h(s, a)
        var(a) = var_h  Q_h(s, a)
        base_score(a) = mu(a) - lambda_var * var(a)
        score(a)      = base_score(a) + HF(t, a)

    Each head is updated independently with standard Q-learning, optionally
    adding Gaussian noise to the TD-error for a subset of heads.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        n_heads: int = 5,
        gamma: float = 0.99,
        alpha: float = 0.1,
        A: float = 0.5,
        omega: float = 0.1,
        phase_offset: float = 0.0,
        lambda_var: float = 0.5,
        prior: Optional[np.ndarray] = None,
        tau: float = 1.0,
        name: str = "LevoThinking",
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_heads = n_heads
        self.gamma = gamma
        self.alpha = alpha
        self.A = A
        self.omega = omega
        self.phase_offset = phase_offset
        self.lambda_var = lambda_var
        self.prior = prior
        self.tau = tau

        self.name = name
        self.Q = np.zeros((n_heads, n_states, n_actions), dtype=float)
        self.t = 0  # global step counter for the HF modulation

    def select_action(self, state: int, rng: np.random.Generator) -> int:
        probs = ensemble_levo_thinking_policy(
            self.Q,
            state,
            t=self.t,
            A=self.A,
            omega=self.omega,
            phase_offset=self.phase_offset,
            lambda_var=self.lambda_var,
            prior=self.prior,
            tau=self.tau,
        )
        self.t += 1
        return int(rng.choice(self.n_actions, p=probs))

    def update(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        done: bool,
        noisy_heads: Optional[Iterable[int]] = None,
        noise_std: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Update each head with standard Q-learning.

        If `noisy_heads` is not None and `noise_std > 0`, then for any head h
        in `noisy_heads`, we add N(0, noise_std) to the TD-error before the
        update. This is used in Phase 3 to simulate corrupted learning signals.
        """
        if noisy_heads is None:
            noisy_set = set()
        else:
            noisy_set = set(int(h) for h in noisy_heads)

        for h in range(self.n_heads):
            q_sa = self.Q[h, s, a]
            if done:
                target = r
            else:
                target = r + self.gamma * float(self.Q[h, s_next].max())
            td = target - q_sa

            if h in noisy_set and noise_std > 0.0 and rng is not None:
                td += float(rng.normal(loc=0.0, scale=noise_std))

            self.Q[h, s, a] = q_sa + self.alpha * td
