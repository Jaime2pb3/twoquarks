

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.append(PROJECT_ROOT)

import os
import csv
import numpy as np
from envs.hypothesis_lab_env import HypothesisLabEnv
from levo.strange_swarm import StrangeConfig, StrangeSwarm


def run_experiment(
    n_episodes: int = 200,
    max_steps: int = 50,
    seed: int = 0,
    out_path: str = "../results/strange_hypothesis_lab.csv",
):
    env = HypothesisLabEnv(max_steps=max_steps, seed=seed)
    cfg = StrangeConfig(n_actions=env.n_actions, seed=seed)
    swarm = StrangeSwarm(cfg)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = [
        "episode", "step", "reward", "T", "H_swarm", "S_t", "diversity",
        "w_t", "phase", "progress", "stability"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(n_episodes):
            s = env.reset()
            done = False
            step = 0
            while not done:
                a, info = swarm.act(s, step)
                s_next, r, done, env_info = env.step(a)
                swarm.update(s, a, r, s_next, done)
                row = {
                    "episode": ep,
                    "step": step,
                    "reward": r,
                    "T": info["T"],
                    "H_swarm": info["H_swarm"],
                    "S_t": info["S_t"],
                    "diversity": info["diversity"],
                    "w_t": info["w_t"],
                    "phase": env_info["phase"],
                    "progress": env_info["progress"],
                    "stability": env_info["stability"],
                }
                writer.writerow(row)
                s = s_next
                step += 1


if __name__ == "__main__":
    run_experiment()
