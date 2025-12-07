
import csv
from pathlib import Path

import numpy as np

try:
    from .env_paradox import EpistemicValleyEnv  # type: ignore
    from .agent_levo_paradox import HFLevoAgent, LevoParadoxIsomerAgent  # type: ignore
except ImportError:  # script-style usage
    from env_paradox import EpistemicValleyEnv
    from agent_levo_paradox import HFLevoAgent, LevoParadoxIsomerAgent


def run_experiment(
    out_csv: str | Path,
    episodes_per_phase: int = 400,
    seed: int = 2025,
) -> None:
    """
    Run HF-Levo and Levo Paradox tabular agents and log results to CSV.

    The CSV schema is:

        global_episode,phase,episode,env_seed,agent,episode_reward,failure_mode,rho_state
    """
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # env for shape information only
    probe_env = EpistemicValleyEnv(phase=1, seed=seed)
    n_states = probe_env.n_states
    n_actions = probe_env.n_actions

    agents = [
        HFLevoAgent(n_states, n_actions, seed=seed),
        LevoParadoxIsomerAgent(n_states, n_actions, seed=seed + 1),
    ]
    agent_names = ["HFLevo", "LevoParadoxIsomer"]

    global_ep = 0

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

                for agent, name in zip(agents, agent_names, strict=True):
                    env = EpistemicValleyEnv(phase=phase, seed=env_seed)
                    s_idx, info = env.reset()
                    a = agent.select_action(s_idx)
                    s_next_idx, reward, done, step_info = env.step(a)

                    failure_mode = step_info.get("failure_mode", "neutral")

                    rho_state = (
                        float(getattr(agent, "rho")[s_idx])  # type: ignore[attr-defined]
                        if hasattr(agent, "rho")
                        else float("nan")
                    )

                    agent.update(
                        s_idx=s_idx,
                        a=a,
                        r=reward,
                        s_next_idx=s_next_idx,
                        done=done,
                        failure_mode=failure_mode,
                    )

                    writer.writerow(
                        {
                            "global_episode": global_ep,
                            "phase": phase,
                            "episode": local_ep,
                            "env_seed": env_seed,
                            "agent": name,
                            "episode_reward": float(reward),
                            "failure_mode": failure_mode,
                            "rho_state": rho_state,
                        }
                    )

                global_ep += 1


if __name__ == "__main__":
    run_experiment("results/paradox_tabular_results.csv", episodes_per_phase=400)
