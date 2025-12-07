# Anti-CHARM v2 — Contextual Risk Regularizer

## 1. Role within the Quarks Architecture

Anti-CHARM is the antiquark of the CHARM module. It does not seek to optimize
reward directly; its purpose is to **limit the effect of enchanted valleys and
contextual cycles** that can overpower the alignment mechanisms of a main agent.

The effective decision is computed as:

\[
Q_{\text{eff}}(s,a) = Q_{\text{charm}}(s,a) - \lambda_t \cdot P_{\text{anti}}(s,a)
\]

where:

- `Q_charm(s,a)`: value proposed by the “enchanted” actor (short path / reward).
- `P_anti(s,a)`: contextual risk penalty estimated by Anti-CHARM.
- `\lambda_t`: dynamic isomeric gain in the range `[lambda_min, lambda_max]`.

When the context “heats up” (concentrated reward, low diversity),
`\lambda_t` increases and Anti-CHARM gains influence. In cold and varied
contexts, `\lambda_t` decreases and CHARM dominates.

---

## 2. Risk Components

Anti-CHARM decomposes risk into three analytic terms:

### 2.1 Loop Risk (`loop_risk`)

A state graph is constructed from observed transitions.
Floyd–Warshall is applied on this graph to estimate the minimum cost of
returning to the same state `s`:

\[
loop\_cost(s) = dist(s,s)
\]

This cost is normalized via `tanh`:

\[
loop\_risk(s) = \tanh\left( \frac{loop\_cost(s)}{10} \right)
\]

Short, low-cost cycles result in higher risk.

---

### 2.2 Valley Score (`valley_score`)

For each aggregated edge `(s, a, s')`, statistics are kept for:

- mean reward `\bar r(s,a)`,
- mean progress toward the goal `\bar p(s,a)`.

Enchanted valleys correspond to **high reward with low progress**:

\[
valley\_{raw}(s,a) = \max(0, \bar r(s,a) - \bar p(s,a))
\]

Normalized:

\[
valley\_score(s,a) = \tanh(valley\_{raw}(s,a))
\]

---

### 2.3 Context Pressure (`context_pressure`)

Measures reward and visitation concentration around state `s`:

\[
context\_pressure(s) = \tanh\Bigg(
    \frac{\sum_{a,s'} R(s,a,s')}{\sum_{a,s'} N(s,a,s') + \epsilon}
\Bigg)
\]

---

### 2.4 Base Penalty

\[
P_{\text{base}}(s,a) =
    \alpha_{loop}\,loop\_risk(s) +
    \beta_{valley}\,valley\_score(s,a) +
    \gamma_{context}\,context\_pressure(s)
\]

with configurable weights `alpha_loop`, `beta_valley`, and `gamma_context`.

---

## 3. Learned Risk Calibrator

To avoid rigidity in the analytic penalty, Anti-CHARM includes a small
supervised **risk calibrator** `RiskCalibrator`.

Per-step input features:

- `p_base` — analytic base penalty.
- `step_norm` — normalized step index.
- `H_policy` — policy entropy.
- `temp` — effective temperature.
- `diversity` — trajectory/action diversity.
- `mean_reward_ep` — episode mean reward.

Output:

- `delta_penalty` — scalar correction to the base risk.

Final penalty:

\[
P_{\text{anti}}(s,a) = P_{\text{base}}(s,a) + \Delta P_{\theta}(s,a)
\]

---

### 3.1 Explicit Risk Target

For each episode:

- **Episode diversity**  
  \(diversity\_{ep} = \frac{|\{s_t\}|}{T}\).

- **Loop rate**  
  \(loop\_rate = 1 - diversity\_{ep}\).

- **Diversity gap**  
  positive difference from `diversity_target`.

- **Reward collapse**  
  \(reward\_collapse =
     \frac{\sigma(r_t)}{|\bar r_t| + \epsilon}\).

The **risk label** is:

\[
risk\_label = \tanh(loop\_rate + diversity\_gap + reward\_collapse)
\]

This label is used as the regression target:

\[
\mathcal{L} = \mathbb{E}[(\Delta P_\theta - risk\_label)^2]
\]

The calibrator therefore adapts penalties when loops, low diversity, or reward
collapse states are detected.

---

## 4. Analytic Isomeric Gain λ_t

\(\lambda_t\) is computed analytically from two global signals:

- reward density `reward_density`,
- current `diversity`.

Score:

\[
score = \tanh(reward\_density) + (diversity\_{target} - diversity)
\]

Saturated to `[-1,1]` and mapped to `[lambda_min, lambda_max]`:

\[
\lambda_t = mid + span \cdot score
\]

Result:

- Hot, low-diversity contexts ⟹ `\lambda_t` approaches `lambda_max`.
- Cold, diverse contexts ⟹ `\lambda_t` approaches `lambda_min`.

This procedure is transparent and stable by design.

---

## 5. Weak Points and Mitigations

1. **Floyd–Warshall complexity O(n³)**  
   - Mitigation: `max_fw_states` and sparse refresh scheduling
     (`fw_interval_episodes`).

2. **Insufficient data states**  
   - Mitigation: mild default penalties and near-zero context pressure prevent
     disproportionate punishment.

3. **Over-penalization (excessive paranoia)**  
   - Mitigation: risk terms are bounded by `tanh`, and risk weights are
     configurable.

4. **Coarse risk labels**  
   - Mitigation: although labels are episode-level, step-wise features allow
     the calibrator to learn finer intra-episode patterns.

---

## 6. Integration with CHARM

1. CHARM computes `Q_charm(s,a)`.
2. Anti-CHARM computes `P_anti(s,a)` and `λ_t` via `penalty_vector`.
3. The effective policy uses:

   \[
   Q_{\text{eff}}(s,a) = Q_{\text{charm}}(s,a) - \lambda_t P_{\text{anti}}(s,a)
   \]

4. The agent selects `argmax_a Q_eff(s,a)` or samples from `softmax(Q_eff)`.

Anti-CHARM thus acts as a **contextual regulator** that reshapes the decision
landscape rather than censoring specific actions, reducing the attractiveness
of enchanted valleys and loops.