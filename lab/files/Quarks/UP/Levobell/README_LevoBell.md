# LevoBell — structural reward-driven intelligence

Tabular Q-learning with structural modulation and reward shaping.

- Environment: 5x5 GridWorld, slip 0.15, reward -0.01 / +1.
- Baselines:
  - Bellman-eps (epsilon-greedy).
  - Bellman-Reflex-Q (softmax policy).
- LevoBell:
  - Modulation of temperature, entropy weight, learning assertiveness and cadence.
  - Low-pass policy filter and jerk/entropy based shaping of the reward signal.
- Metrics:
  - Success rate, steps, normalised entropy H, jerk KL, ECE, Brier score.
  - Wilson confidence intervals and two-proportion z-test versus Bellman-Reflex-Q.

Experiments: 50 x 1000 episodes by default in `run_all`, configurable in the function parameters.
