
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover - torch might not be installed
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    optim = object  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tabular baselines (HF-Levo + isomeric Levo Paradox)
# ---------------------------------------------------------------------------


class HFLevoAgent:
    """
    Single-head HF-Levo baseline agent.

    A simple Q-learning agent whose exploration temperature is modulated
    by a high-frequency term:

        T_t = tau * (1 + A * sin(omega * t))

    This creates oscillatory exploration / exploitation cycles.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        A: float = 0.5,
        omega: float = 0.05,
        tau: float = 0.7,
        alpha: float = 0.1,
        gamma: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.A = A
        self.omega = omega
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma

        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def _temperature(self) -> float:
        return float(self.tau * (1.0 + self.A * math.sin(self.omega * self.t)))

    def _softmax(self, q: np.ndarray, temp: float) -> np.ndarray:
        z = q - np.max(q)
        e = np.exp(z / max(temp, 1e-6))
        return e / e.sum()

    def select_action(self, s_idx: int) -> int:
        temp = self._temperature()
        probs = self._softmax(self.Q[s_idx], temp)
        a = int(self.rng.choice(self.n_actions, p=probs))
        self.t += 1
        return a

    def update(
        self,
        s_idx: int,
        a: int,
        r: float,
        s_next_idx: int,
        done: bool,
        failure_mode: str,
    ) -> None:
        q = self.Q
        target = r if done else (r + self.gamma * float(np.max(q[s_next_idx])))
        delta = target - float(q[s_idx, a])
        q[s_idx, a] += self.alpha * delta


class LevoParadoxIsomerAgent:
    """
    Isomeric tabular agent with conservative and aggressive Q-functions.

    Two heads Q_L (conservative) and Q_R (aggressive) are mixed by a
    contextual polarization rho[s] in [0, 1]:

        Q_mix[s] = (1 - rho[s]) * Q_L[s] + rho[s] * Q_R[s]

    Polarization is nudged by failure modes reported by the environment:
      - "overconfident"  -> push rho down (more conservative)
      - "overcautious"   -> push rho up   (more aggressive)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        tau: float = 0.7,
        alpha: float = 0.1,
        gamma: float = 0.95,
        eta: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta

        self.Q_L = np.zeros((n_states, n_actions), dtype=np.float32)
        self.Q_R = np.zeros((n_states, n_actions), dtype=np.float32)
        self.rho = np.full(n_states, 0.5, dtype=np.float32)

        self.rng = np.random.default_rng(seed)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        e = np.exp(z / max(self.tau, 1e-6))
        return e / e.sum()

    def select_action(self, s_idx: int) -> int:
        rho_s = float(self.rho[s_idx])
        q_mix = (1.0 - rho_s) * self.Q_L[s_idx] + rho_s * self.Q_R[s_idx]
        probs = self._softmax(q_mix)
        a = int(self.rng.choice(self.n_actions, p=probs))
        return a

    def update(
        self,
        s_idx: int,
        a: int,
        r: float,
        s_next_idx: int,
        done: bool,
        failure_mode: str,
    ) -> None:
        # choose learning head: conservative for overconfidence,
        # aggressive for overcautious, otherwise both share credit.
        if failure_mode == "overconfident":
            heads = ("L",)
        elif failure_mode == "overcautious":
            heads = ("R",)
        else:
            heads = ("L", "R")

        for h in heads:
            Q = self.Q_L if h == "L" else self.Q_R
            target = r if done else (r + self.gamma * float(np.max(Q[s_next_idx])))
            delta = target - float(Q[s_idx, a])
            Q[s_idx, a] += self.alpha * delta

        # polarization update
        if failure_mode == "overconfident":
            self.rho[s_idx] = np.clip(self.rho[s_idx] - self.eta, 0.0, 1.0)
        elif failure_mode == "overcautious":
            self.rho[s_idx] = np.clip(self.rho[s_idx] + self.eta, 0.0, 1.0)


# ---------------------------------------------------------------------------
# PPO Hybrid Engine (GPU-ready, isomeric actor-critic)
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    state_idx: int
    action: int
    reward: float
    log_prob: float
    value: float
    rho: float
    failure_mode: str


class ParadoxActorCritic(nn.Module):  # type: ignore[misc]
    """
    Isomeric actor-critic.

    - shared MLP trunk over a one-hot encoding of the discrete state
    - two actor heads: conservative vs aggressive
    - one scalar gating head producing rho(s) in [0, 1]
    - one critic head V(s)

    The final policy is a mixture of the two isomers, combined inside the
    logits space and fed through softmax.
    """

    def __init__(self, n_states: int, n_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.trunk = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_cons = nn.Linear(hidden_dim, n_actions)
        self.actor_aggr = nn.Linear(hidden_dim, n_actions)
        self.gate = nn.Linear(hidden_dim, 1)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state_one_hot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.trunk(state_one_hot)
        logits_cons = self.actor_cons(z)
        logits_aggr = self.actor_aggr(z)
        rho = torch.sigmoid(self.gate(z))  # [B, 1]
        value = self.critic(z).squeeze(-1)  # [B]

        # mixture in logit space
        logits_mix = (1.0 - rho) * logits_cons + rho * logits_aggr
        return logits_mix, value, rho.squeeze(-1), logits_cons * 1.0  # last term unused but can help debugging


class LevoParadoxPPOHybrid:
    """
    GPU-ready PPO hybrid engine with isomeric policy.

    This agent combines:
      - an isomeric actor (conservative + aggressive heads),
      - a gating network rho(s) trained end-to-end,
      - PPO-style clipped policy updates,
      - and a value baseline for variance reduction.

    It operates directly on the discrete Epistemic Valley state space using
    a one-hot encoding, which keeps things simple and fully reproducible.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        lr: float = 3e-4,
        batch_size: int = 256,
        update_epochs: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for LevoParadoxPPOHybrid.")

        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size
        self.update_epochs = update_epochs

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ParadoxActorCritic(n_states, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.buffer: List[Transition] = []

    # ---------------------------- utilities -----------------------------

    def _one_hot(self, idx: np.ndarray | int) -> torch.Tensor:
        idx_arr = np.atleast_1d(idx).astype(np.int64)
        x = np.zeros((idx_arr.shape[0], self.n_states), dtype=np.float32)
        x[np.arange(idx_arr.shape[0]), idx_arr] = 1.0
        return torch.from_numpy(x).to(self.device)

    # ---------------------------- interaction ---------------------------

    def select_action(self, s_idx: int) -> Tuple[int, float, float, float]:
        """Return (action, log_prob, value_estimate, rho)."""
        self.net.eval()
        state_one_hot = self._one_hot(s_idx)
        logits_mix, value, rho, _ = self.net(state_one_hot)  # type: ignore[misc]
        dist = torch.distributions.Categorical(logits=logits_mix)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item()), float(rho.item())

    def store_transition(
        self,
        s_idx: int,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        rho: float,
        failure_mode: str,
    ) -> None:
        self.buffer.append(
            Transition(
                state_idx=int(s_idx),
                action=int(action),
                reward=float(reward),
                log_prob=float(log_prob),
                value=float(value),
                rho=float(rho),
                failure_mode=failure_mode,
            )
        )

    # ---------------------------- learning ------------------------------

    def _compute_advantages(
        self, rewards: np.ndarray, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # For one-step episodes, advantage = reward - value, return = reward
        returns = rewards.copy()
        advantages = rewards - values
        # Normalise for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self) -> Dict[str, float]:
        if not self.buffer:
            return {}

        # Collect batch
        states = np.array([t.state_idx for t in self.buffer], dtype=np.int64)
        actions = np.array([t.action for t in self.buffer], dtype=np.int64)
        rewards = np.array([t.reward for t in self.buffer], dtype=np.float32)
        old_log_probs = np.array([t.log_prob for t in self.buffer], dtype=np.float32)
        values = np.array([t.value for t in self.buffer], dtype=np.float32)

        returns, advantages = self._compute_advantages(rewards, values)

        # Convert to tensors
        states_t = self._one_hot(states)
        actions_t = torch.from_numpy(actions).to(self.device)
        returns_t = torch.from_numpy(returns).to(self.device)
        advantages_t = torch.from_numpy(advantages).to(self.device)
        old_log_probs_t = torch.from_numpy(old_log_probs).to(self.device)

        dataset_size = states_t.size(0)
        idxs = np.arange(dataset_size)

        stats: Dict[str, float] = {}

        self.net.train()
        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = idxs[start : start + self.batch_size]
                if len(batch_idx) == 0:
                    continue

                batch_states = states_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_adv = advantages_t[batch_idx]
                batch_old_logp = old_log_probs_t[batch_idx]

                logits_mix, values_pred, _, _ = self.net(batch_states)  # type: ignore[misc]
                dist = torch.distributions.Categorical(logits=logits_mix)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_logp)
                unclipped = ratio * batch_adv
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = (batch_returns - values_pred).pow(2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

                stats = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }

        # clear buffer
        self.buffer.clear()
        return stats
