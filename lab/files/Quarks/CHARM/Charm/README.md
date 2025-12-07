# CHARM — Enchanted Valley

This module implements the **CHARM** quark over the Enchanted Valley environment.

- The environment is a non-stationary graph with multiple routes and shifting
  risk phases.
- `CharmField` builds a structural field from the explored subgraph.
- A tabular Q-learning agent uses this field as potential-based shaping, coupled
  via λ, which is tuned using a Lion-style sign–momentum optimizer.

Main file:

- `levo/charm_enchanted_valley.py` — includes:
    - the Enchanted Valley environment definition,
    - the `CharmField` class,
    - the `CharmAgent`,
    - the function `train_charm_enchanted_valley(...)` that trains the agent and
      returns statistics ready for plotting.

Notebook workflow:

`notebooks/Charm_EnchantedValley_core.ipynb`

1. Set strategic hyperparameters (episodes, γ, initial λ, etc.).
2. Run `train_charm_enchanted_valley`.
3. Visualize reward evolution, λ dynamics, and average polarization ρ̄ per episode.
