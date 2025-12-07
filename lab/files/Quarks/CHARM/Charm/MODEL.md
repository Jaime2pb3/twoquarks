# CHARM — Structured Field Model in Enchanted Valley

## 1. Role inside the Quarks Architecture

CHARM is the quark focused on **exploiting structure** in environments with rich
topology and non-stationary dynamics. It does not modify the reward objective,
but instead deforms the decision landscape through a learned potential field over
the explored graph.

The Enchanted Valley environment is modeled as a directed graph with:

- “short” routes that become risky during certain phases,
- longer but safer alternative paths,
- phase shifts that modify the risk of specific edges.

CHARM introduces a potential field Φ(s) and a coupling coefficient λ such that
the effective policy of the tabular agent becomes:

\[
\pi(a\mid s) \propto
\exp\bigl(Q(s,a) + \lambda \cdot \Delta \Phi(s,a)\bigr)
\]

where:

- `Q(s,a)` is the tabular TD value estimate,
- `ΔΦ(s,a)` is the potential difference estimated by `CharmField`,
- `λ` is adjusted through a meta-control step (Lion optimizer) based on observed
  structural gain.

---

## 2. Internal Signals

CHARM maintains several internal signals:

- `edge_costs(s,s')` — estimated structural cost for each edge (derived from reward
  signals and TD-error),
- `V(s)` and `ρ(s)` — local critic value and **isomeric polarization** extracted from
  TD-errors,
- `diffusion_field(s)` — K-step diffusion field propagated from goal states,
- `lambda_field` — current intensity of structural shaping.

These terms are recombined to produce potential fields that bias the agent towards
robust trajectories under changing valley phases.

---

## 3. Recommended Usage

- Study reward evolution with Charm's field active vs disabled.
- Observe the interaction between λ and environmental non-stationarity.
- Use Enchanted Valley as a controlled *testbed* for structural shaping ideas
  before scaling to larger environments.