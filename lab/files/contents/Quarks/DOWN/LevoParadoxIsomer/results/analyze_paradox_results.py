
"""
Simple analysis helpers for Levo Paradox experiments.

This module is intentionally minimal: it loads the CSV produced by
`run_paradox_tabular.py` or `run_paradox_ppo_hybrid.py` and prints
aggregate statistics, and can optionally produce a couple of matplotlib
plots if the library is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib might be absent
    plt = None  # type: ignore[assignment]


def load_results(path: str | Path) -> Dict[str, np.ndarray]:
    import csv

    path = Path(path)
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    def col(name: str, cast):
        return np.array([cast(r[name]) for r in rows])

    return {
        "global_episode": col("global_episode", int),
        "phase": col("phase", int),
        "episode": col("episode", int),
        "agent": np.array([r["agent"] for r in rows]),
        "reward": col("episode_reward", float),
        "failure_mode": np.array([r["failure_mode"] for r in rows]),
        "rho_state": col("rho_state", float),
    }


def print_summary(path: str | Path) -> None:
    data = load_results(path)
    agents = np.unique(data["agent"])
    phases = np.unique(data["phase"])

    print(f"Summary for {path}:")
    for agent in agents:
        mask_agent = data["agent"] == agent
        print(f"\nAgent: {agent}")
        print(f"  Mean reward (all): {data['reward'][mask_agent].mean():.3f}")
        for phase in phases:
            mask = mask_agent & (data["phase"] == phase)
            if mask.sum() == 0:
                continue
            mean_r = data["reward"][mask].mean()
            print(f"  Phase {phase}: mean reward = {mean_r:.3f}")


def plot_reward_curve(path: str | Path, rolling: int = 50) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    data = load_results(path)

    agents = np.unique(data["agent"])
    fig, ax = plt.subplots()
    for agent in agents:
        mask = data["agent"] == agent
        rewards = data["reward"][mask]
        episodes = data["global_episode"][mask]
        if len(rewards) < rolling:
            rolling_rewards = rewards
            ep = episodes
        else:
            rolling_rewards = np.convolve(
                rewards, np.ones(rolling) / rolling, mode="valid"
            )
            ep = episodes[rolling - 1 :]
        ax.plot(ep, rolling_rewards, label=agent)

    ax.set_xlabel("Global episode")
    ax.set_ylabel(f"Reward (rolling {rolling})")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print_summary(args.csv_path)
    if args.plot:
        plot_reward_curve(args.csv_path)
