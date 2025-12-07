# Epistemic Valley — Isomeric RL under Ambiguity

This repository contains a small but fully reproducible RL lab built around
the **Epistemic Valley** environment: a discrete decision problem that
models how an agent behaves under *ambiguity, risk and changing reward
regimes*.

The focus is not on reaching perfect reward, but on **exposing how different
inductive biases behave under stress**:

- **HFLevo** – a high‑frequency Levo‑style tabular agent whose exploration
  temperature oscillates over time.
- **LevoParadoxIsomer** – a tabular *isomeric* agent with conservative and
  aggressive Q‑functions mixed by a contextual polarization ρ(s).
- **LevoParadoxPPOHybrid** – a neural PPO‑style hybrid actor–critic used
  as a **diagnostic probe** to show how large‑capacity methods behave on a
  tiny, one‑step epistemic MDP.

The lab is designed to be easy to run end‑to‑end on a laptop or GPU box and
to make the trade‑offs visible in a couple of minutes.

---

## 1. Environment — Epistemic Valley Inframe v2

The environment is implemented in `env_paradox.py` as a one‑step MDP with:

- **State** `s = (ambiguity_type, evidence_bin, risk_level, phase)`  
  where
  - `ambiguity_type ∈ {lexical, scope, missing‑data}`
  - `evidence_bin ∈ {low, medium, high}`
  - `risk_level ∈ {low, high}`
  - `phase ∈ {1, 2, 3}` (reward regime)
- **Actions**
  - `HOLD`   – maintain the current position.
  - `EXPAND` – seek more evidence or broader context.
  - `CLARIFY`– request targeted clarification.
  - `DEFER`  – postpone the decision.
  - `ANSWER` – commit to an answer now.

Episodes are single‑step: `reset() -> step(action) -> done=True`.  This keeps
analysis simple and makes the asymmetries between agents very visible.

The reward function encodes three intuitive failure modes:

- **Overconfidence** – taking aggressive actions under high risk + low evidence.
- **Overcautiousness** – refusing to answer under high evidence + low risk.
- **Neutral** – everything else, with milder shaping.

Phases change how forgiving the environment is:

1. Phase 1 – slightly optimistic, small bonus for aligned aggression.  
2. Phase 2 – harsher penalties for overconfidence.  
3. Phase 3 – noisy feedback (Gaussian noise, clipped), mimicking ambiguous,
   unstable supervision.

See `env_paradox.py` and `MODELS.md` for the exact specification.

---

## 2. Agents

All agents share the same interface but have different inductive biases.

### 2.1 HFLevo — Oscillatory Baseline

`HFLevoAgent` in `agent_levo_paradox.py` is a tabular Q‑learning agent.
Its exploration temperature oscillates over time:

- `T_t = τ (1 + A sin(ω t))`

This creates exploration / exploitation cycles without storing extra state.
On Epistemic Valley it serves as a **strong but simple baseline**.

Empirically:

- Global mean reward: **0.538**
- Phase 1: **0.731**
- Phase 2: **0.486**
- Phase 3: **0.398**

HFLevo performs well in the forgiving regime and degrades gracefully as
penalties and noise increase.

### 2.2 LevoParadoxIsomer — Tabular Isomeric Policy

`LevoParadoxIsomerAgent` implements the core idea of **isomeric RL**:

- two Q‑tables:
  - `Q_L` – conservative
  - `Q_R` – aggressive
- a learned polarization `ρ(s) ∈ [0, 1]` that mixes them:

      Q_mix(s) = (1 − ρ(s)) Q_L(s) + ρ(s) Q_R(s)

- polarization is nudged by failure mode:
  - overconfidence → push ρ(s) down (more conservative)
  - overcautiousness → push ρ(s) up (more aggressive)

Empirically:

- Global mean reward: **0.546**
- Phase 1: **0.741**
- Phase 2: **0.399**
- Phase 3: **0.498**

The pattern is informative:

- In Phase 1 the isomer and HFLevo behave similarly.
- In Phase 2 (harsh penalties) HFLevo retains a slight edge.
- In Phase 3 (noisy feedback) the **isomeric agent is clearly more robust,
  improving mean reward by ~0.10 over HFLevo.**

The isomer does not dominate everywhere; instead it **specialises in ambiguous,
noisy regimes**, which is exactly the behaviour it was designed to explore.

### 2.3 LevoParadoxPPOHybrid — Neural Probe

`LevoParadoxPPOHybrid` in `agent_levo_paradox.py` is a PyTorch actor–critic
agent with:

- a shared MLP trunk over one‑hot states;
- conservative and aggressive policy heads mixed by a learned ρ(s);
- a value head and PPO‑style clipped updates;
- automatic CPU / CUDA selection:

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

On Epistemic Valley the PPO hybrid is deliberately used as a **diagnostic
probe**, not as a performance benchmark: the environment is tiny (54 states,
one‑step episodes), which structurally favours tabular methods.

Empirically:

- Global mean reward: **0.256**
- Phase 1: **0.266**
- Phase 2: **0.182**
- Phase 3: **0.320**

The PPO hybrid underperforms both tabular agents here.  That is not a bug but
a **signal**: high‑capacity neural methods are not automatically superior on
small, sharply‑shaped decision surfaces.  In this lab the PPO engine plays the
role of *“strong generalist out of domain”* — a useful contrast to your
specialised tabular experts.

---

## 3. Running the lab

From inside the project directory:

```bash
# Tabular baselines
python run_paradox_tabular.py

# PPO hybrid probe (uses GPU if available)
python run_paradox_ppo_hybrid.py
```

Both scripts write their logs under `results/`:

- `results/paradox_tabular_results.csv`
- `results/paradox_ppo_results.csv`

Basic analysis and plotting:

```bash
# Numeric summaries
python analyze_paradox_results.py results/paradox_tabular_results.csv
python analyze_paradox_results.py results/paradox_ppo_results.csv

# Rolling reward curves (needs matplotlib)
python analyze_paradox_results.py results/paradox_tabular_results.csv --plot
python analyze_paradox_results.py results/paradox_ppo_results.csv --plot
```

The repository already includes **one full run** of each script plus the
generated plots, so you can inspect the behaviour without re‑training.

---

## 4. Files

- `env_paradox.py` – Epistemic Valley environment (one‑step epistemic MDP).
- `agent_levo_paradox.py` – HFLevo, LevoParadoxIsomer and LevoParadoxPPOHybrid.
- `run_paradox_tabular.py` – experiment runner for the tabular agents.
- `run_paradox_ppo_hybrid.py` – experiment runner for the PPO hybrid probe.
- `analyze_paradox_results.py` – utilities for loading, summarising and
  plotting results.
- `results/paradox_tabular_results.csv` – logged run for HFLevo + isomer.
- `results/paradox_ppo_results.csv` – logged run for the PPO hybrid.
- `results/Figure_1.png` – rolling reward for HFLevo vs LevoParadoxIsomer.
- `results/Figure_2.png` – rolling reward for LevoParadoxPPOHybrid.
- `01_paradox_tabular.ipynb` – notebook for interactive exploration.

---

## 5. How to read this lab

This project is intentionally **small, opinionated and imperfect**.  The goal
is not to present a single “best” agent, but to expose how different models
shine under different conditions:

- The **oscillatory HFLevo** is a strong generalist on clean regimes.
- The **isomeric agent** is more robust when feedback is noisy and ambiguous.
- The **PPO hybrid** reminds us that capacity alone does not guarantee
  performance on small, structured problems.

If you extend the environment to multi‑step episodes or larger state spaces,
the same agents will behave differently.  That is the real value of Epistemic
Valley: a controlled arena where you can see inductive biases collide.
