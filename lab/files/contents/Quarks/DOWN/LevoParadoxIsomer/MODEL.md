Epistemic Valley — Models
==========================

1. Environment: Epistemic Valley Inframe v2
-------------------------------------------

We model a one‑step Markov decision process (MDP) intended to capture how an
agent behaves when faced with ambiguous evidence and varying risk.

State

- `s = (ambiguity_type, evidence_bin, risk_level, phase)`
- `ambiguity_type ∈ {lexical, scope, missing‑data}`
- `evidence_bin ∈ {low, medium, high}`
- `risk_level ∈ {low, high}`
- `phase ∈ {1, 2, 3}`

There are 3×3×2×3 = 54 discrete states.  Each phase defines a different reward
regime.

Actions

- `0 = HOLD`
- `1 = EXPAND`
- `2 = CLARIFY`
- `3 = DEFER`
- `4 = ANSWER`

Episodes are single‑step, so each episode consists of:

- sample `s` from a phase‑specific distribution;
- apply an action `a`;
- observe reward `r(s, a)` and a failure mode label.

Reward and failure modes

Let `e` be the evidence bin, `r` the risk level and `φ` the phase.

- aggressive actions are `{EXPAND, ANSWER}`
- conservative actions are `{HOLD, DEFER}`

Core cases:

- **Overconfidence** (failure_mode = "overconfident"):
  - high risk and low evidence with an aggressive action
  - base reward around −2 with additional penalties in Phase 2
- **Overcautiousness** (failure_mode = "overcautious"):
  - low risk and high evidence with a conservative action
  - moderate negative reward (≈ −0.8)
- **Aligned behaviour**:
  - aggressive + high evidence + low risk → positive reward (up to ≈ +2.5)
  - conservative + high risk + low evidence → mild positive reward

Phase shaping:

1. Phase 1 – mildly optimistic: successful aggressive moves get a small bonus.
2. Phase 2 – harsher penalties for overconfidence.
3. Phase 3 – additive Gaussian noise, clipped to a fixed range, modelling
   unstable supervision.

The exact implementation lives in `_reward` inside `EpistemicValleyEnv` in
`env_paradox.py`.

2. HFLevo: Oscillatory Q‑learning
---------------------------------

HFLevo is a tabular Q‑learning agent with a temperature schedule

- `T_t = τ (1 + A sin(ω t))`

At each decision time `t` it samples an action from

- `π(a | s) = softmax(Q(s, ·) / T_t)`

and updates Q with

- `Q(s, a) ← Q(s, a) + α (r + γ max_{a'} Q(s', a') − Q(s, a))`

The oscillatory temperature drives alternating phases of exploration and
exploitation without introducing extra state into the policy.

3. LevoParadoxIsomer: Tabular Isomeric Agent
--------------------------------------------

The isomeric agent maintains two Q‑tables:

- `Q_L(s, a)` – conservative
- `Q_R(s, a)` – aggressive

and a per‑state polarization `ρ(s) ∈ [0, 1]`.

Action selection

For each state `s` we form a mixed value:

- `Q_mix(s, a) = (1 − ρ(s)) Q_L(s, a) + ρ(s) Q_R(s, a)`

and sample actions from

- `π(a | s) = softmax(Q_mix(s, ·) / τ)`

Learning

Given a transition `(s, a, r, s', done, failure_mode)` we update:

- choose heads depending on failure mode:
  - if `failure_mode = "overconfident"`: update **only** `Q_L`
  - if `failure_mode = "overcautious"`: update **only** `Q_R`
  - otherwise: update both
- for each updated head `Q`:

  - target `y = r` if `done`, else `r + γ max_{a'} Q(s', a')`
  - `Q(s, a) ← Q(s, a) + α (y − Q(s, a))`

Polarization dynamics

- if `failure_mode = "overconfident"`:

  - `ρ(s) ← clip(ρ(s) − η, 0, 1)`

- if `failure_mode = "overcautious"`:

  - `ρ(s) ← clip(ρ(s) + η, 0, 1)`

This ties the bias of the mixture to observed epistemic failures.  Over time
the agent tends to be more conservative in states that often trigger
overconfidence and more aggressive where excessive caution is harmful.

4. LevoParadoxPPOHybrid: Neural Probe
-------------------------------------

The PPO hybrid agent is a neural actor–critic used as a **probe** on this
small MDP.

Architecture

- Input: one‑hot encoding of the 54 states.
- Trunk: two‑layer MLP with ReLU activations.
- Heads:
  - conservative actor logits `z_L(s)`
  - aggressive actor logits `z_R(s)`
  - gating scalar `g(s)` with `ρ(s) = σ(g(s))`
  - value head `V(s)`

The mixed policy is

- `z_mix(s) = (1 − ρ(s)) z_L(s) + ρ(s) z_R(s)`
- `π(a | s) = softmax(z_mix(s))`

PPO objective

Given batches of transitions collected on‑policy, we form

- `r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`
- clipped policy objective

  `L^π(θ) = E_t[ min( r_t(θ) Â_t, clip(r_t(θ), 1 − ε, 1 + ε) Â_t ) ]`

where for this one‑step environment the advantages reduce to

- `Â_t = r_t − V_θ(s_t)` (with normalisation).

We add a value loss

- `L^V(θ) = E_t[(R_t − V_θ(s_t))^2]`

and an entropy bonus to discourage premature collapse.  The final loss is

- `L(θ) = −L^π(θ) + c_v L^V(θ) − c_e H[π_θ]`

with Adam optimisation.

On Epistemic Valley this neural agent underperforms the tabular ones; this is
expected and documented, and makes it a useful contrastive tool rather than a
primary benchmark.
