from typing import Optional

import numpy as np

from .policies import hf_levo_policy_tabular


class LevoQTabularAgent:
    """
    Tabular Q-learning agent with an HF-Levo policy.

    The Q-update is standard:
        Q(s, a) <- Q(s, a) + alpha * (r + gamma max_a' Q(s', a') - Q(s, a))

    The policy is:
        score(a) = Q(s, a) + A cos(omega * t + phi_a + phase_offset)
        pi(a|s)  = softmax(score / tau)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        A: float = 0.5,
        omega: float = 0.1,
        phase_offset: float = 0.0,
        ent_weight: float = 0.0,
        prior: Optional[np.ndarray] = None,
        tau: float = 1.0,
        name: str = "LevoQ",
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.A = A
        self.omega = omega
        self.phase_offset = phase_offset
        self.ent_weight = ent_weight
        self.prior = prior
        self.tau = tau

        self.name = name
        self.Q = np.zeros((n_states, n_actions), dtype=float)
        self.t = 0  # global step counter for the HF modulation

    def select_action(self, state: int, rng: np.random.Generator) -> int:
        probs = hf_levo_policy_tabular(
            self.Q,
            state,
            t=self.t,
            A=self.A,
            omega=self.omega,
            phase_offset=self.phase_offset,
            ent_weight=self.ent_weight,
            prior=self.prior,
            tau=self.tau,
        )
        self.t += 1
        return int(rng.choice(self.n_actions, p=probs))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        q_sa = self.Q[s, a]
        if done:
            target = r
        else:
            target = r + self.gamma * float(self.Q[s_next].max())
        td = target - q_sa
        self.Q[s, a] = q_sa + self.alpha * td
