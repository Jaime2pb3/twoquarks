Epistemic Valley — Graphics
===========================

All plots in this lab are generated from the CSV logs under `results/`.

1. Figure_1 — Tabular Agents (HFLevo vs Isomer)
-----------------------------------------------

File: `results/Figure_1.png`

- Rolling‑window reward (window = 50 episodes) vs global episode.
- Blue curve: **HFLevo**.
- Orange curve: **LevoParadoxIsomer**.

The figure shows:

- Similar performance in the early, forgiving regime (Phase 1).
- Diverging behaviour as phases change: the isomer dips more in Phase 2 but
  recovers and stabilises better once noisy rewards dominate (Phase 3).
- The two agents are not simple rescalings of each other; they represent
  different epistemic strategies.

2. Figure_2 — PPO Hybrid Probe
------------------------------

File: `results/Figure_2.png`

- Rolling‑window reward for **LevoParadoxPPOHybrid** alone.

The curve exhibits:

- pronounced oscillations and drops, especially around phase transitions;
- lower overall reward than both tabular agents;
- a clear example of how a high‑capacity neural policy can struggle on a
  tiny, sharply shaped MDP where tabular methods are structurally advantaged.

3. Re‑creating the plots
------------------------

To regenerate the figures from scratch:

```bash
python run_paradox_tabular.py
python run_paradox_ppo_hybrid.py

python analyze_paradox_results.py results/paradox_tabular_results.csv --plot
python analyze_paradox_results.py results/paradox_ppo_results.csv --plot
```

You can also extend `analyze_paradox_results.py` to compute additional
statistics (e.g. per‑phase variance, failure‑mode frequencies, or correlations
between ρ(s) and reward) if you want to dig deeper into the behaviour of each
agent.
