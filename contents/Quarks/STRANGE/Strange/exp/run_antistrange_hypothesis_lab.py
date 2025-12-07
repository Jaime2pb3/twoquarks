import os
import sys
import csv
import numpy as np

# --- Ajustar ruta raíz del proyecto ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.append(PROJECT_ROOT)

# OJO: la clase AntiHypothesisLabEnv vive en hypothesis_lab_env.py
from envs.hypothesis_lab_env import AntiHypothesisLabEnv
from levo.antistrange_swarm import AntiStrangeConfig, AntiStrangeSwarm


def run_antistrange(
    n_episodes: int = 200,
    max_steps: int = 50,
    seed: int = 0,
    out_path: str = "../results/antistrange_hypothesis_lab.csv",
):
    # Entorno "territorio anti" 🤘
    env = AntiHypothesisLabEnv(max_steps=max_steps, seed=seed)
    cfg = AntiStrangeConfig(n_actions=env.n_actions, seed=seed)
    agent = AntiStrangeSwarm(cfg)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = [
        "episode", "step", "reward",
        "T", "H_swarm", "S_t", "diversity", "w_t",
        "phase", "progress", "stability",
        "latent_potential",  # NUEVO: clave para ver cuándo despierta el anti
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(n_episodes):
            s = env.reset()
            done = False
            step = 0

            while not done:
                a, info = agent.act(s, step)
                s_next, r, done, env_info = env.step(a)
                agent.update(s, a, r, s_next, done)

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
                    "latent_potential": env_info.get("latent_potential", 0.0),
                }
                writer.writerow(row)

                s = s_next
                step += 1


if __name__ == "__main__":
    run_antistrange()
