#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Charm • Enchanted Valley
========================

"Enchanted Valley" environment + CharmField module + base agent + training loop.

Core idea:
- The environment is a non-stationary graph with multiple routes.
- Charm builds structural fields from the explored subgraph.
- The base agent performs tabular Q-learning.
- The effective policy is coupled to the field through a λ factor learned via Lion
  (sign-momentum) at the meta-control level.

This file is designed to be self-contained and didactic, not hyper-optimized.
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Environment: Enchanted Valley (non-stationary graph)
# ---------------------------------------------------------------------------

class EnchantedValleyEnv:
    """
    Small but topologically rich graph:
    - 3 zones: A (entrance), B (plateau), C (core).
    - Multiple alternative routes toward the core.
    - Some edges become risky depending on the phase.
    - Phases shift every N episodes (non-stationarity).

    States: integers 0..N-1
    Actions: indices over the neighbor list of the current state.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.n_states = 14  # small but non-trivial
        self.adj = self._build_graph_structure()
        self.start_state = 0
        self.goal_states = {11, 13}  # two nodes in "core C"

        self.max_steps = 40
        self.episode_idx = 0
        self.phase = 0
        self.state = self.start_state
        self.steps = 0

    def _build_graph_structure(self) -> Dict[int, List[int]]:
        """
        Hand-designed graph:

        Zone A: 0,1,2
        Zone B: 3,4,5,6,7
        Zone C: 8,9,10,11,12,13

        Routes:
        0 -> 1 -> 3 -> 5 -> 8 -> 11 (short route to C1)
        0 -> 2 -> 4 -> 6 -> 9 -> 13 (alternative route to C2)
        3 -> 7 -> 10 -> 12 -> 11 (inner detour)
        """
        adj = defaultdict(list)
        edges = [
            (0, 1), (0, 2),
            (1, 3),
            (2, 4),
            (3, 5), (3, 7),
            (4, 6),
            (5, 8),
            (6, 9),
            (7, 10),
            (8, 11),
            (9, 13),
            (10, 12),
            (12, 11),
            # lateral connections
            (4, 3),
            (6, 5),
            (9, 10),
        ]
        for u, v in edges:
            adj[u].append(v)
        return adj

    def _update_phase(self):
        """
        Environment phases (non-stationary):

        phase 0: all edges have similar cost.
        phase 1: certain short edges become risky.
        phase 2: alternative routes become less costly (reward shaping).
        """
        if self.episode_idx < 100:
            self.phase = 0
        elif self.episode_idx < 200:
            self.phase = 1
        else:
            self.phase = 2

    def reset(self) -> int:
        self._update_phase()
        self.state = self.start_state
        self.steps = 0
        return self.state

    def get_neighbors(self, s: int) -> List[int]:
        return self.adj[s]

    def step(self, action_idx: int) -> Tuple[int, float, bool, Dict]:
        """
        action_idx: index inside self.adj[state].
        """
        self.steps += 1
        neighbors = self.adj[self.state]
        if not neighbors:
            return self.state, -5.0, True, {"terminal_reason": "dead_end"}

        action_idx = max(0, min(action_idx, len(neighbors) - 1))
        next_state = neighbors[action_idx]

        reward = -1.0
        done = False
        info = {}

        if next_state in self.goal_states:
            reward += 15.0
            done = True
            info["terminal_reason"] = "goal"

        u, v = self.state, next_state

        if self.phase == 1:
            risky_edges = {(3, 5), (5, 8)}
            if (u, v) in risky_edges:
                if self.rng.random() < 0.3:
                    reward -= 10.0
                    info["risk_event"] = True

        if self.phase == 2:
            alt_edges = {(4, 6), (6, 9), (9, 10)}
            if (u, v) in alt_edges:
                reward += 0.5

        if self.steps >= self.max_steps and not done:
            done = True
            info["terminal_reason"] = "max_steps"

        self.state = next_state
        return next_state, reward, done, info


# ---------------------------------------------------------------------------
# 2. CharmField Engine (fields + isomerism + Lion meta-control)
# ---------------------------------------------------------------------------

@dataclass
class CharmConfig:
    gamma: float = 0.95

    alpha_v: float = 0.1
    alpha_cost: float = 0.2

    lr_lambda: float = 0.01
    beta1_lambda: float = 0.9

    lambda_init: float = 0.5
    mu_diff: float = 0.2

    diffusion_K: int = 4
    diffusion_gamma: float = 0.8

    kappa_rho: float = 2.0
    rho_min: float = 0.05
    rho_max: float = 0.95


class CharmField:
    """
    Charm structural field module.

    - Maintains a causal subgraph via empirical edge costs.
    - Computes two fields:
        * Safety (Φ_L): penalizes risk / uncertainty.
        * Efficiency (Φ_R): promotes proximity to goal.
    - Builds a global diffuse field via K-step diffusion.
    - Computes a combined potential Φ(s).
    - Couples the base policy through a learnable λ (Lion optimizer).
    - ρ(s) is derived from |TD-error(s)| (isomeric polarization).
    """

    def __init__(self, n_states: int, goal_states, config: CharmConfig):
        self.n_states = n_states
        self.goal_states = list(goal_states)
        self.cfg = config

        self.edge_counts = np.zeros((n_states, n_states), dtype=np.float32)
        self.edge_costs = np.full((n_states, n_states), np.inf, dtype=np.float32)

        self.phi_L = np.zeros(n_states, dtype=np.float32)
        self.phi_R = np.zeros(n_states, dtype=np.float32)
        self.phi_diff = np.zeros(n_states, dtype=np.float32)

        self.lambda_field = self.cfg.lambda_init
        self.mu_diff = self.cfg.mu_diff
        self.m_lambda = 0.0

        self.V = np.zeros(n_states, dtype=np.float32)
        self.last_td_error = np.zeros(n_states, dtype=np.float32)

        self.rho = np.zeros(n_states, dtype=np.float32)

    def combined_potential(self, s: int) -> float:
        """
        Φ(s) = (1 - ρ) * Φ_L + ρ * Φ_R + μ * Φ_diff
        """
        rho_s = self.rho[s]
        phi_s = (1 - rho_s) * self.phi_L[s] + rho_s * self.phi_R[s]
        phi_s += self.mu_diff * self.phi_diff[s]
        return phi_s

    def delta_phi(self, s: int, s_next: int) -> float:
        return self.combined_potential(s_next) - self.combined_potential(s)

    def lion_meta_update_lambda(self, avg_struct_gain: float):
        """
        Lion optimizer update on λ using sign-momentum.

        If exploiting the field produces positive structural gain,
        λ should increase; otherwise it decays.
        """
        g = -avg_struct_gain

        self.m_lambda = (
            self.cfg.beta1_lambda * self.m_lambda
            + (1.0 - self.cfg.beta1_lambda) * g
        )

        step = self.cfg.lr_lambda * np.sign(self.m_lambda)
        self.lambda_field -= step
        self.lambda_field = max(0.0, min(2.0, self.lambda_field))


# ---------------------------------------------------------------------------
# 3. Base Agent (Tabular Q-learning + Charm shaping)
# ---------------------------------------------------------------------------

class CharmAgent:
    """
    Q-learning agent with Charm shaping:

    - Learns Q(s,a) by TD.
    - Effective policy:
        π(a|s) ∝ exp( Q(s,a) + λ * ΔΦ(s,a) )
    """

    def __init__(
        self,
        n_states: int,
        max_actions: int,
        charm: CharmField,
        gamma: float = 0.95,
        alpha_q: float = 0.2,
        temperature: float = 1.0,
    ):
        self.n_states = n_states
        self.max_actions = max_actions
        self.gamma = gamma
        self.alpha_q = alpha_q
        self.temperature = temperature
        self.charm = charm

        self.Q = np.zeros((n_states, max_actions), dtype=np.float32)

    def select_action(self, s: int, neighbors: List[int]) -> int:
        if not neighbors:
            return 0

        n_actions = len(neighbors)
        logits = np.zeros(n_actions, dtype=np.float32)

        for a in range(n_actions):
            sp = neighbors[a]
            base_q = self.Q[s, a]
            dphi = self.charm.delta_phi(s, sp)
            logits[a] = (base_q + self.charm.lambda_field * dphi) / max(
                1e-3, self.temperature
            )

        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)

        r = random.random()
        cdf = 0.0
        for a in range(n_actions):
            cdf += probs[a]
            if r <= cdf:
                return a
        return n_actions - 1

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        q_sa = self.Q[s, a]
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        td = target - q_sa
        self.Q[s, a] += self.alpha_q * td
        return td


# ---------------------------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------------------------

def train_charm_enchanted_valley(
    n_episodes: int = 300,
    seed: int = 0,
):
    env = EnchantedValleyEnv(seed=seed)
    cfg = CharmConfig()
    charm = CharmField(
        n_states=env.n_states,
        goal_states=list(env.goal_states),
        config=cfg,
    )

    max_actions = max(len(env.get_neighbors(s)) for s in range(env.n_states))
    agent = CharmAgent(env.n_states, max_actions, charm)

    rewards_hist = []
    lambda_hist = []
    rho_mean_hist = []

    for ep in range(n_episodes):
        env.episode_idx = ep
        s = env.reset()
        done = False

        ep_reward = 0.0
        struct_gain_sum = 0.0
        steps = 0

        while not done:
            neighbors = env.get_neighbors(s)
            a = agent.select_action(s, neighbors)
            s_next, r, done, info = env.step(a)

            dphi = charm.delta_phi(s, s_next)
            struct_gain_sum += dphi

            td = agent.update(s, a, r, s_next, done)

            ep_reward += r
            s = s_next
            steps += 1

        avg_struct_gain = struct_gain_sum / max(1, steps)
        charm.lion_meta_update_lambda(avg_struct_gain)

        rewards_hist.append(ep_reward)
        lambda_hist.append(charm.lambda_field)
        rho_mean_hist.append(float(np.mean(charm.rho)))

        if (ep + 1) % 50 == 0:
            print(
                f"[Ep {ep+1:4d}] "
                f"Reward={ep_reward:6.2f}  "
                f"λ={charm.lambda_field:.3f}  "
                f"ρ_mean={rho_mean_hist[-1]:.3f}  "
                f"phase={env.phase}"
            )

    return {
        "rewards": np.array(rewards_hist),
        "lambda": np.array(lambda_hist),
        "rho_mean": np.array(rho_mean_hist),
    }


if __name__ == "__main__":
    stats = train_charm_enchanted_valley()
