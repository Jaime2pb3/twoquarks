# Levo-HF: Corrupted Valley (tabular)

Minimal tabular RL experiment on a non-stationary gridworld with a corrupted valley
and two Levo-style agents (HF-Levo and LevoThinking).

## Layout

- `envs/corrupted_valley.py` – 7x7 gridworld with a corrupted valley and three phases.
- `levo/levo_q_tabular.py` – HF-Levo tabular Q-learning agent.
- `levo/levo_thinking_ensemble.py` – LevoThinking ensemble Q-learning agent.
- `levo/policies.py` – shared HF and ensemble policies.
- `exp/run_corrupted_valley_tabular.py` – training script.
- `exp/plot_corrupted_valley_results.py` – plotting script.
- `utils/logging.py` – minimal CSV logger.

## Requirements

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install numpy matplotlib
