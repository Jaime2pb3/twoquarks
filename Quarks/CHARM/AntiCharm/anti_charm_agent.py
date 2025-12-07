"""
Anti-Charm Agent v2
--------------------
Robust contextual regularizer for CHARM-like agents.

Idea:
    Q_eff(s,a) = Q_charm(s,a) - lambda_t * P_anti(s,a)

where P_anti is a risk penalty that estimates:
    - loopiness (getting stuck in local cycles),
    - fake-valley attraction (high reward / low progress),
    - contextual pressure (reward + repetition concentrated in one area),
  and lambda_t is a bounded gain that grows when the context "heats up"
  (concentrated reward, low diversity) and decays when the system is cold.

Assumes:
    - Discrete and relatively small state space (e.g., gridworlds).
    - Scalar progress in [0,1] (0 = far from goal, 1 = goal).
    - Basic per-step stats: policy entropy, temperature, diversity.

This module does NOT decide the optimal action by itself.
It deforms the decision landscape so that the main actor (CHARM)
is less vulnerable to enchanted valleys and contextual loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW


@dataclass
class AntiCharmConfig:
    num_states: int
    num_actions: int

    # Analytic weights for the base penalty
    alpha_loop: float = 1.0      # loop risk weight
    beta_valley: float = 1.0     # fake-valley weight
    gamma_context: float = 0.5   # contextual pressure weight

    # Global targets
    diversity_target: float = 0.6
    max_fw_states: int = 512          # hard cap for Floyd–Warshall
    fw_interval_episodes: int = 5     # how often to refresh the graph

    # Risk calibrator training
    calib_lr: float = 1e-3
    weight_decay: float = 1e-3

    # Lambda range (isomeric gain)
    lambda_min: float = 0.05
    lambda_max: float = 0.95

    device: str = "cpu"


class RiskCalibrator(nn.Module):
    """
    Small MLP that adjusts the base penalty using an explicit risk target.

    Input : feature vector (6,)
    Output: scalar delta_penalty
    """
    def __init__(self, in_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim)
        return self.net(x).squeeze(-1)  # (batch,)


class AntiCharmAgent:
    """
    Implements the CHARM anti-quark (ANTI-CHARM) as a contextual risk module.

    Expected usage (pseudocode):

        anti = AntiCharmAgent(cfg)

        for episode in range(N):
            s = env.reset()
            done = False
            step = 0
            while not done:
                # 1) main actor proposes values
                Q_charm = charm.q_values(s)      # shape: (num_actions,)

                # 2) Anti-Charm computes per-action penalties
                stats = {...}  # H_policy, temp, diversity, reward_density, ...
                P_anti, lambda_t = anti.penalty_vector(s, stats)  # shape: (num_actions,)

                Q_eff = Q_charm - lambda_t * P_anti
                a = int(Q_eff.argmax())

                s_next, r, done, info = env.step(a)

                # 3) log transition for Anti-Charm
                anti.observe_step(
                    s=s, a=a, r=r, s_next=s_next,
                    progress=info.get("progress", 0.0),
                    H_policy=stats.get("H_policy", 0.0),
                    temp=stats.get("temp", 1.0),
                    diversity=stats.get("diversity", 0.0),
                    episode_id=episode,
                    step=step,
                    done=done,
                )

                s = s_next
                step += 1

            anti.end_episode()

    """

    def __init__(self, cfg: AntiCharmConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        n_s = cfg.num_states
        n_a = cfg.num_actions

        # Per-edge transition statistics (s,a,s')
        self.edge_counts = np.zeros((n_s, n_a, n_s), dtype=np.int32)
        self.edge_reward = np.zeros((n_s, n_a, n_s), dtype=np.float32)
        self.edge_progress = np.zeros((n_s, n_a, n_s), dtype=np.float32)
        self.edge_steps = np.zeros((n_s, n_a, n_s), dtype=np.float32)

        # Distances for Floyd–Warshall (computed on collapsed graph s -> s')
        self.dist = np.full((n_s, n_s), np.inf, dtype=np.float32)
        np.fill_diagonal(self.dist, 0.0)

        # Episode buffers
        self.current_episode_steps: List[Dict[str, Any]] = []
        self.episode_counter: int = 0

        # Risk calibrator
        self.calibrator = RiskCalibrator(in_dim=6).to(self.device)
        self.opt_calib = AdamW(
            self.calibrator.parameters(),
            lr=cfg.calib_lr,
            weight_decay=cfg.weight_decay,
        )

    # ------------------------------------------------------------------
    #  ONLINE MONITORING
    # ------------------------------------------------------------------
    def observe_step(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        progress: float,
        H_policy: float,
        temp: float,
        diversity: float,
        episode_id: int,
        step: int,
        done: bool,
    ) -> None:
        """
        Register a transition (s, a, r, s') together with contextual stats.
        """
        # update edge statistics
        self._update_edge_stats(s, a, s_next, r, progress)

        # buffer for episode-level training
        self.current_episode_steps.append(
            dict(
                s=s,
                a=a,
                r=r,
                s_next=s_next,
                progress=progress,
                H_policy=H_policy,
                temp=temp,
                diversity=diversity,
                step=step,
                done=done,
            )
        )

    def _update_edge_stats(self, s: int, a: int, s_next: int, r: float, progress: float) -> None:
        """
        Accumulate statistics for edge (s, a, s_next).
        """
        self.edge_counts[s, a, s_next] += 1
        self.edge_reward[s, a, s_next] += float(r)
        self.edge_progress[s, a, s_next] += float(progress)
        self.edge_steps[s, a, s_next] += 1.0

    # ------------------------------------------------------------------
    #  ONLINE PENALTY
    # ------------------------------------------------------------------
    def penalty_vector(self, s: int, stats: Dict[str, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the penalty vector P_anti(s, a) for all actions and the current
        lambda_t. Does not update any parameters.
        """
        n_a = self.cfg.num_actions
        p_vec = np.zeros((n_a,), dtype=np.float32)

        # compute lambda_t from global contextual stats
        lambda_t = self._compute_lambda_analytic(stats)

        for a in range(n_a):
            p_vec[a] = self._penalty_single_action(s, a, stats)

        return p_vec, float(lambda_t)

    def _penalty_single_action(self, s: int, a: int, stats: Dict[str, float]) -> float:
        """
        Total penalty for a specific action in a given state.

        P_anti = P_base + delta_calib
        """
        # analytic base
        p_base, features = self._compute_penalty_base(s, a)

        # features for the risk calibrator
        feat_vec = np.array(
            [
                p_base,
                stats.get("step_norm", 0.0),
                stats.get("H_policy", 0.0),
                stats.get("temp", 1.0),
                stats.get("diversity", 0.0),
                stats.get("reward_density", 0.0),
            ],
            dtype=np.float32,
        )
        x = torch.tensor(feat_vec, device=self.device).unsqueeze(0)
        with torch.no_grad():
            delta = self.calibrator(x).item()

        return float(p_base + delta)

    # ------------------------------------------------------------------
    #  BASE PENALTY P_base
    # ------------------------------------------------------------------
    def _refresh_distances_if_needed(self) -> None:
        """
        Runs Floyd–Warshall on the collapsed graph s->s' if:
          - the number of states is manageable, and
          - enough episodes have passed.
        """
        cfg = self.cfg
        if cfg.num_states > cfg.max_fw_states:
            # graph too large: keep dist as identity / inf
            return

        # build transition cost matrix (s -> s')
        n_s = cfg.num_states
        cost = np.full((n_s, n_s), np.inf, dtype=np.float32)
        np.fill_diagonal(cost, 0.0)

        for s in range(n_s):
            for sp in range(n_s):
                counts_sa = self.edge_counts[s, :, sp].sum()
                if counts_sa == 0:
                    continue
                mean_r = self.edge_reward[s, :, sp].sum() / max(counts_sa, 1)
                mean_prog = self.edge_progress[s, :, sp].sum() / max(counts_sa, 1)

                # cost: 1 step + penalty for "suspiciously high" reward
                # with low progress (enchanted valleys)
                valley_term = max(0.0, mean_r - mean_prog)
                step_cost = 1.0 + valley_term
                cost[s, sp] = min(cost[s, sp], step_cost)

        # classic Floyd–Warshall
        dist = cost.copy()
        for k in range(n_s):
            dist = np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])

        self.dist = dist

    def _compute_penalty_base(self, s: int, a: int) -> Tuple[float, Dict[str, float]]:
        """
        Combines three components:

            - loop_risk(s):  short cycles s -> ... -> s with low cost.
            - valley_score(s,a): high reward / low progress.
            - context_pressure(s): accumulated reward + visits around s.

        Output is passed through tanh to avoid explosions.
        """
        cfg = self.cfg

        # loop risk: cheapest observed cycle starting and ending at s
        loop_cost = float(self.dist[s, s])
        loop_risk = math.tanh(loop_cost / 10.0)  # soft normalization

        # stats for the specific edge (s,a,*)
        counts = self.edge_counts[s, a, :].sum()
        if counts == 0:
            # no data: mild but non-infinite penalty
            valley_score = 0.1
            context_pressure = 0.1
        else:
            mean_r = self.edge_reward[s, a, :].sum() / counts
            mean_prog = self.edge_progress[s, a, :].sum() / counts
            visits_from_s = self.edge_counts[s, :, :].sum()
            total_reward_from_s = self.edge_reward[s, :, :].sum()

            valley_raw = max(0.0, mean_r - mean_prog)
            valley_score = math.tanh(valley_raw)

            if visits_from_s > 0:
                context_pressure = math.tanh(
                    (total_reward_from_s / visits_from_s)
                )
            else:
                context_pressure = 0.0

        p_base = (
            cfg.alpha_loop * loop_risk
            + cfg.beta_valley * valley_score
            + cfg.gamma_context * context_pressure
        )

        features = dict(
            loop_risk=loop_risk,
            valley_score=valley_score,
            context_pressure=context_pressure,
        )
        return float(p_base), features

    # ------------------------------------------------------------------
    #  ANALYTIC LAMBDA
    # ------------------------------------------------------------------
    def _compute_lambda_analytic(self, stats: Dict[str, float]) -> float:
        """
        Computes lambda_t analytically from global stats:

            - Increases when:
                * reward_density is high,
                * diversity is low.

            - Decreases when the system is varied and cold.
        """
        cfg = self.cfg

        reward_density = stats.get("reward_density", 0.0)
        diversity = stats.get("diversity", 0.0)

        # smooth normalization
        r_term = math.tanh(reward_density)
        d_term = (cfg.diversity_target - diversity)

        # approximate score in [-1, 1]
        raw = r_term + d_term
        score = max(-1.0, min(1.0, raw))

        # map to [lambda_min, lambda_max]
        mid = (cfg.lambda_min + cfg.lambda_max) / 2.0
        span = (cfg.lambda_max - cfg.lambda_min) / 2.0
        lam = mid + span * score
        return float(max(cfg.lambda_min, min(cfg.lambda_max, lam)))

    # ------------------------------------------------------------------
    #  EPISODIC TRAINING OF THE CALIBRATOR
    # ------------------------------------------------------------------
    def end_episode(self) -> None:
        """
        Must be called at the end of each episode.

        - Updates the graph (at a given episode interval).
        - Trains the risk calibrator towards an explicit risk target.

        Episode-level risk target mixes:
            - loop_rate       (repeated states / steps)
            - diversity_gap   (target diversity - actual diversity, if positive)
            - reward_collapse (high reward std relative to its magnitude)
        """
        if not self.current_episode_steps:
            return

        self.episode_counter += 1

        steps = self.current_episode_steps
        num_steps = len(steps)

        states = [s["s"] for s in steps]
        rewards = [s["r"] for s in steps]
        diversities = [s["diversity"] for s in steps]

        unique_states = len(set(states))
        diversity_ep = unique_states / max(num_steps, 1)

        # state repetition rate (loop_rate)
        loop_rate = 1.0 - diversity_ep

        # gap vs. diversity target
        diversity_gap = max(0.0, self.cfg.diversity_target - diversity_ep)

        # reward collapse: std relative to mean magnitude
        mean_r = float(np.mean(rewards)) if num_steps > 0 else 0.0
        std_r = float(np.std(rewards)) if num_steps > 0 else 0.0
        reward_collapse = std_r / (abs(mean_r) + 1e-6)

        # global risk label for the episode (bounded)
        risk_label = math.tanh(loop_rate + diversity_gap + reward_collapse)

        # batch for the calibrator
        X_list: List[np.ndarray] = []
        Y_list: List[float] = []

        for st in steps:
            p_base, _ = self._compute_penalty_base(st["s"], st["a"])
            feat_vec = np.array(
                [
                    p_base,
                    st["step"] / max(num_steps - 1, 1),
                    st["H_policy"],
                    st["temp"],
                    st["diversity"],
                    mean_r,
                ],
                dtype=np.float32,
            )
            X_list.append(feat_vec)
            Y_list.append(risk_label)

        X = torch.tensor(np.stack(X_list, axis=0), device=self.device)
        y = torch.tensor(np.array(Y_list, dtype=np.float32), device=self.device)

        # simple regression training
        self.opt_calib.zero_grad()
        pred = self.calibrator(X)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        self.opt_calib.step()

        # refresh graph distances every few episodes
        if self.episode_counter % self.cfg.fw_interval_episodes == 0:
            self._refresh_distances_if_needed()

        # clear buffer
        self.current_episode_steps = []
