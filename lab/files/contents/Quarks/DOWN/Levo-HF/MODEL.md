Levo-HF / Corrupted Valley – Models
===================================

Environment
-----------

- 7x7 gridworld.
- Start state S at (row=6, col=0).
- Goal state G at (row=0, col=6).
- A 3x3 "valley" region centered in the grid.

Reward structure:

- Each step incurs a small time penalty `r_step < 0`.
- Reaching the goal yields `r_goal = +1.0`.
- Entering the valley yields:

  - Phase 1: `r_valley = r_step + r_phase1`, with `r_phase1 > 0` (falsely attractive).
  - Phase 2: `r_valley = r_step + r_phase2`, with `r_phase2 <= 0` (corrected).
  - Phase 3: same as Phase 2. Noise is injected in the learning signals of the
    LevoThinking ensemble, not in the environment.

Episodes terminate on reaching the goal or after a fixed horizon `T_max`.

State representation:

- Integer index `s in {0, ..., 48}`, encoding (row, col) in row-major order.

Actions:

- 0 = up, 1 = right, 2 = down, 3 = left (clipped to the grid boundaries).


Baseline tabular Q-learning
---------------------------

For each agent, we maintain a table `Q(s, a)`.

The update rule is standard:

- Let

  - `s`  = current state,
  - `a`  = action,
  - `r`  = reward,
  - `s'` = next state,
  - `gamma` in (0, 1] = discount,
  - `alpha` in (0, 1] = learning rate.

- Target:

  `target = r` if the transition is terminal, otherwise

  `target = r + gamma * max_{a'} Q(s', a')`.

- Update:

  `Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))`.

Two baselines are used:

- EpsGreedyQAgent: epsilon-greedy exploration on `Q(s, a)`.
- SoftmaxBoltzmannAgent: Boltzmann exploration with temperature `tau`.


HF-Levo (tabular)
-----------------

HF-Levo keeps a single Q-table, but its policy is modulated in time by a
high-frequency cosine term.

Let `phi_a` denote an action-specific phase, spaced uniformly on `[0, 2π)`:

- `phi_a = 2π * a / |A|`, where `|A|` is the number of actions.

At time step `t`, the score for action `a` in state `s` is:

- `score(a) = Q(s, a) + A * cos(omega * t + phi_a + phi_0)`

where

- `A`      = modulation amplitude,
- `omega`  = angular frequency,
- `phi_0`  = global phase offset.

The policy is a softmax over the scores:

- `pi(a | s, t) = softmax_a(score(a) / tau)`.

Optionally, a prior distribution `pi_0(a)` can be mixed in with weight
`lambda in [0, 1]`:

- `pi_mix(a | s, t) = (1 - lambda) * pi(a | s, t) + lambda * pi_0(a)`.


LevoThinking ensemble
---------------------

LevoThinking maintains an ensemble of `H` Q-tables:

- `Q_h(s, a)` for `h in {1, ..., H}`.

For a given state `s`, define:

- `mu(a)  = (1 / H) * sum_h Q_h(s, a)`
- `var(a) = (1 / H) * sum_h (Q_h(s, a) - mu(a))^2`

The ensemble score is:

- `base_score(a) = mu(a) - lambda_var * var(a)`

where `lambda_var >= 0` penalizes actions whose estimated value is
inconsistent across heads.

HF modulation is then applied as in HF-Levo:

- `score(a) = base_score(a) + A * cos(omega * t + phi_a + phi_0)`

and the policy is again:

- `pi(a | s, t) = softmax_a(score(a) / tau)`.

Each head is updated independently with the tabular Q-learning rule.

In Phase 3, for a subset of heads `H_noisy` and noise level `sigma > 0`,
the TD-error is perturbed by additive Gaussian noise:

- `delta_h <- delta_h + epsilon_h`,
- `epsilon_h ~ Normal(0, sigma^2)` for `h in H_noisy`.

The ensemble policy still aggregates all heads, making the behavior
robust to a fraction of corrupted learning signals.