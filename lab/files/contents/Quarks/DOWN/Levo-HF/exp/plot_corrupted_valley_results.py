"""
Plot results for the Corrupted Valley tabular experiment.

Expects a CSV file created by `exp/run_corrupted_valley_tabular.py` with columns:
    phase, phase_tag, episode, agent, total_reward, valley_visits
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


RESULTS_CSV = os.path.join("results", "corrupted_valley_tabular.csv")
OUT_PREFIX = os.path.join("results", "corrupted_valley_")


def load_results(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_global_index(rows: List[Dict[str, str]]) -> Tuple[Dict[int, int], Dict[int, Dict[str, List[Tuple[int, float, float]]]]]:
    """
    Returns:
        phase_to_max_ep: phase -> max episode index
        data: phase -> agent -> list of (global_ep, total_reward, valley_visits)
    """
    phase_to_eps: Dict[int, List[int]] = defaultdict(list)
    for r in rows:
        phase = int(r["phase"])
        ep = int(r["episode"])
        phase_to_eps[phase].append(ep)

    phase_to_max_ep: Dict[int, int] = {
        phase: max(eps) + 1 for phase, eps in phase_to_eps.items()
    }

    data: Dict[int, Dict[str, List[Tuple[int, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in rows:
        phase = int(r["phase"])
        agent = r["agent"]
        ep = int(r["episode"])
        total_reward = float(r["total_reward"])
        valley_visits = float(r["valley_visits"])
        max_ep = phase_to_max_ep[phase]
        global_ep = (phase - 1) * max_ep + ep
        data[phase][agent].append((global_ep, total_reward, valley_visits))

    return phase_to_max_ep, data


def plot_metric(
    data: Dict[int, Dict[str, List[Tuple[int, float, float]]]],
    metric_index: int,
    ylabel: str,
    out_path: str,
) -> None:
    """
    metric_index: 1 for total_reward, 2 for valley_visits.
    """
    plt.figure()

    # merge across phases, keeping breaks in the global index
    agent_to_xy: Dict[str, Tuple[List[int], List[float]]] = {}

    for phase, agents in sorted(data.items()):
        for agent, triples in agents.items():
            xs = [t[0] for t in triples]
            ys = [t[metric_index] for t in triples]
            if agent not in agent_to_xy:
                agent_to_xy[agent] = ([], [])
            agent_to_xy[agent][0].extend(xs)
            agent_to_xy[agent][1].extend(ys)

    for agent, (xs, ys) in agent_to_xy.items():
        # sort by x
        pairs = sorted(zip(xs, ys), key=lambda p: p[0])
        xs_sorted = [p[0] for p in pairs]
        ys_sorted = [p[1] for p in pairs]
        plt.plot(xs_sorted, ys_sorted, label=agent)

    plt.xlabel("global episode index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Results file not found: {RESULTS_CSV}")

    rows = load_results(RESULTS_CSV)
    _, data = build_global_index(rows)

    plot_metric(
        data,
        metric_index=1,
        ylabel="total reward per episode",
        out_path=OUT_PREFIX + "reward.png",
    )

    plot_metric(
        data,
        metric_index=2,
        ylabel="valley visits per episode",
        out_path=OUT_PREFIX + "valley_visits.png",
    )


if __name__ == "__main__":
    main()
