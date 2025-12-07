# Bellman-Shift Stability Baseline

Minimal baseline for temporal-consistency evaluation under Bellman updates.

**Core measures**
- Policy shift:      KL(πₜ || πₜ₋₁)
- Value shift:       mean |Qₜ − Qₜ₋₁|
- Stability index:   exp(−ΔQ)
- Entropy profile:   H(πₜ)
- TD variability:    var(δₜ) grouped by cosine-similarity neighborhoods

**Adaptive step**
αₜ = αΔ · exp(−t / T)

**TD batches**
ΔQₜ = αₜ · δₜ

**Optimizers tested**
- AdamW  
- Lion

**Environment**
5×5 GridWorld with stochastic slip.

All metrics are computed per-iteration and aggregated in rolling windows for stability diagnostics.
