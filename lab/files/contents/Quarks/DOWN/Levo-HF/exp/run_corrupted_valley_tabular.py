"""
Tabular experiments in the Corrupted Valley Gridworld.

Agents:
- EpsGreedyQAgent (baseline)
- SoftmaxBoltzmannAgent
- LevoQTabularAgent
- LevoThinkingEnsembleAgent

Phases:
1) Corrupted reward in the valley (falsely attractive).
2) Corrected reward.
3) Corrected reward + noisy TD-updates for LevoThinking heads.
"""

import os
from typing import Dict, List, Optional, Iterable

import numpy as np

from envs.corrupted_valley import CorruptedValleyEnv
from levo.levo_q_tabular import LevoQTabularAgent
from levo.levo_thinking_ensemble import LevoThinkingEnsembleAgent
from utils.logging import CSVLogger


class EpsGreedyQAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        epsilon: float = 0.1,
        gamma: float = 0.99,
        alpha: float = 0.1,
        name: str = "EpsGreedy",
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.name = name
        self.Q = np.zeros((n_states, n_actions), dtype=float)

    def select_action(self, state: int, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.n_actions))
        q_s = self.Q[state]
        return int(rng.choice(np.flatnonzero(q_s == q_s.max())))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        q_sa = self.Q[s, a]
        if done:
            target = r
        else:
            target = r + self.gamma * float(self.Q[s_next].max())
        td = target - q_sa
        self.Q[s, a] = q_sa + self.alpha * td


class SoftmaxBoltzmannAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        tau: float = 0.5,
        gamma: float = 0.99,
        alpha: float = 0.1,
        name: str = "Softmax",
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.name = name
        self.Q = np.zeros((n_states, n_actions), dtype=float)

    def _softmax(self, q_s: np.ndarray) -> np.ndarray:
        q = q_s / max(self.tau, 1e-8)
        q = q - np.max(q)
        ex = np.exp(q)
        s = ex.sum()
        if s <= 0.0:
            return np.ones_like(ex) / len(ex)
        return ex / s

    def select_action(self, state: int, rng: np.random.Generator) -> int:
        probs = self._softmax(self.Q[state])
        return int(rng.choice(self.n_actions, p=probs))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        q_sa = self.Q[s, a]
        if done:
            target = r
        else:
            target = r + self.gamma * float(self.Q[s_next].max())
        td = target - q_sa
        self.Q[s, a] = q_sa + self.alpha * td


def run_episode(
    env: CorruptedValleyEnv,
    agent,
    phase_id: int,
    episode_idx: int,
    logger: CSVLogger,
    rng: np.random.Generator,
    noisy_heads: Optional[Iterable[int]] = None,
    noise_std: float = 0.0,
) -> None:
    obs = env.reset()
    total_reward = 0.0
    valley_visits = 0
    done = False

    while not done:
        action = agent.select_action(obs, rng=rng)
        step = env.step(action)
        next_obs = step.obs
        reward = step.reward
        done = step.done
        info = step.info

        total_reward += reward
        if info.get("valley", False):
            valley_visits += 1

        if isinstance(agent, LevoThinkingEnsembleAgent):
            agent.update(
                obs,
                action,
                reward,
                next_obs,
                done,
                noisy_heads=noisy_heads,
                noise_std=noise_std,
                rng=rng,
            )
        else:
            agent.update(obs, action, reward, next_obs, done)

        obs = next_obs

    logger.log(
        {
            "phase": phase_id,
            "phase_tag": f"phase{phase_id}",
            "episode": episode_idx,
            "agent": agent.name,
            "total_reward": total_reward,
            "valley_visits": valley_visits,
        }
    )


def run_phase(
    env_seed_base: int,
    env_seed_offset: int,
    phase_id: int,
    n_episodes: int,
    agents: List,
    logger: CSVLogger,
    noisy_heads_spec: Optional[Iterable[int]] = None,
    noise_std: float = 0.0,
) -> None:
    """
    Run one phase for all agents.

    Seeds are constructed as:
        env_seed = env_seed_base + phase_id * env_seed_offset + agent_idx * 10000 + episode_idx
    so that episodes are reproducible but independent across agents and phases.
    """
    for agent_idx, agent in enumerate(agents):
        for ep in range(n_episodes):
            env_seed = (
                env_seed_base
                + phase_id * env_seed_offset
                + agent_idx * 10000
                + ep
            )
            env = CorruptedValleyEnv(seed=env_seed, phase=phase_id)
            rng = np.random.default_rng(env_seed)

            if isinstance(agent, LevoThinkingEnsembleAgent):
                if noisy_heads_spec is None:
                    # default: half of the heads receive noisy TD-updates
                    k = max(1, agent.n_heads // 2)
                    effective_noisy_heads = range(k)
                else:
                    effective_noisy_heads = noisy_heads_spec
                episode_noise_std = noise_std
            else:
                effective_noisy_heads = None
                episode_noise_std = 0.0

            run_episode(
                env=env,
                agent=agent,
                phase_id=phase_id,
                episode_idx=ep,
                logger=logger,
                rng=rng,
                noisy_heads=effective_noisy_heads,
                noise_std=episode_noise_std,
            )


def main() -> None:
    base_seed = 1234
    seed_offset = 1000

    # probe environment to get sizes
    probe_env = CorruptedValleyEnv(seed=base_seed, phase=1)
    n_states = probe_env.n_states
    n_actions = probe_env.n_actions

    agents = [
        EpsGreedyQAgent(n_states, n_actions, epsilon=0.1, name="EpsGreedy"),
        SoftmaxBoltzmannAgent(n_states, n_actions, tau=0.5, name="Softmax"),
        LevoQTabularAgent(
            n_states,
            n_actions,
            A=0.5,
            omega=0.05,
            ent_weight=0.0,
            name="LevoQ",
        ),
        LevoThinkingEnsembleAgent(
            n_states,
            n_actions,
            n_heads=5,
            A=0.5,
            omega=0.05,
            lambda_var=0.5,
            name="LevoThinking",
        ),
    ]

    results_path = os.path.join("results", "corrupted_valley_tabular.csv")
    logger = CSVLogger(
        results_path,
        fieldnames=[
            "phase",
            "phase_tag",
            "episode",
            "agent",
            "total_reward",
            "valley_visits",
        ],
    )

    n_episodes = 200

    # Phase 1: corrupted valley
    run_phase(
        env_seed_base=base_seed,
        env_seed_offset=seed_offset,
        phase_id=1,
        n_episodes=n_episodes,
        agents=agents,
        logger=logger,
        noisy_heads_spec=None,
        noise_std=0.0,
    )

    # Phase 2: corrected valley
    run_phase(
        env_seed_base=base_seed,
        env_seed_offset=seed_offset,
        phase_id=2,
        n_episodes=n_episodes,
        agents=agents,
        logger=logger,
        noisy_heads_spec=None,
        noise_std=0.0,
    )

    # Phase 3: corrected valley + noisy TD-updates for LevoThinking
    run_phase(
        env_seed_base=base_seed,
        env_seed_offset=seed_offset,
        phase_id=3,
        n_episodes=n_episodes,
        agents=agents,
        logger=logger,
        noisy_heads_spec=None,  # default: half of the heads
        noise_std=0.1,
    )


if __name__ == "__main__":
    main()
