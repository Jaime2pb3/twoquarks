
import csv
from pathlib import Path

import numpy as np

try:
    from .env_paradox import EpistemicValleyEnv  # type: ignore
    from .agent_levo_paradox import LevoParadoxPPOHybrid  # type: ignore
except ImportError:
    from env_paradox import EpistemicValleyEnv
    from agent_levo_paradox import LevoParadoxPPOHybrid


def run_ppo_hybrid(
    out_csv: str | Path,
    episodes_per_phase: int = 400,
    seed: int = 2025,
    update_every: int = 256,
) -> None:
    """
    Train the Levo Paradox PPO Hybrid engine on Epistemic Valley and log results.

    We keep the same CSV schema as the tabular runner, plus the PPO-specific
    rho estimate for the visited state.
    """
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # probe env for shapes
    probe_env = EpistemicValleyEnv(phase=1, seed=seed)
    n_states = probe_env.n_states
    n_actions = probe_env.n_actions

    agent = LevoParadoxPPOHybrid(
        n_states=n_states,
        n_actions=n_actions,
        seed=seed,
    )

    global_ep = 0
    steps_since_update = 0

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_episode",
                "phase",
                "episode",
                "env_seed",
                "agent",
                "episode_reward",
                "failure_mode",
                "rho_state",
            ],
        )
        writer.writeheader()

        for phase in (1, 2, 3):
            for local_ep in range(episodes_per_phase):
                env_seed = int(rng.integers(0, 2**31 - 1))
                env = EpistemicValleyEnv(phase=phase, seed=env_seed)

                s_idx, info = env.reset()
                action, logp, value_est, rho_est = agent.select_action(s_idx)
                s_next_idx, reward, done, step_info = env.step(action)
                failure_mode = step_info.get("failure_mode", "neutral")

                agent.store_transition(
                    s_idx=s_idx,
                    action=action,
                    reward=reward,
                    log_prob=logp,
                    value=value_est,
                    rho=rho_est,
                    failure_mode=failure_mode,
                )

                writer.writerow(
                    {
                        "global_episode": global_ep,
                        "phase": phase,
                        "episode": local_ep,
                        "env_seed": env_seed,
                        "agent": "LevoParadoxPPOHybrid",
                        "episode_reward": float(reward),
                        "failure_mode": failure_mode,
                        "rho_state": rho_est,
                    }
                )

                global_ep += 1
                steps_since_update += 1

                if steps_since_update >= update_every:
                    agent.update()
                    steps_since_update = 0

        # final update with any remaining samples
        if steps_since_update > 0:
            agent.update()


if __name__ == "__main__":
    run_ppo_hybrid("results/paradox_ppo_results.csv", episodes_per_phase=400)
