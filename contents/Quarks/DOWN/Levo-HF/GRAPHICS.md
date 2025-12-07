Levo-HF / Corrupted Valley – Graphics
=====================================

The plotting script `exp/plot_corrupted_valley_results.py` generates two figures
from the CSV logs produced by `exp/run_corrupted_valley_tabular.py`.

1. `corrupted_valley_reward.png`

   - x-axis: "global episode index".
     Episodes are concatenated across phases:
       - Phase 1 occupies indices [0, N_episodes_phase1),
       - Phase 2 occupies [N_episodes_phase1, 2 * N_episodes_phase1),
       - Phase 3 occupies [2 * N_episodes_phase1, 3 * N_episodes_phase1).
   - y-axis: "total reward per episode".
   - One line per agent:
       - EpsGreedy
       - Softmax
       - LevoQ
       - LevoThinking
   - Interpretation:
       - Phase 1: a good agent harvests reward from the corrupted valley.
       - Phase 2: a robust agent should unlearn the attraction to the valley and
         reorient toward the true goal.
       - Phase 3: LevoThinking is exposed to noisy TD-updates on a subset of its
         heads; robustness is expressed as a smoother degradation of reward
         compared to non-ensemble agents.

2. `corrupted_valley_valley_visits.png`

   - x-axis: "global episode index" (same convention as above).
   - y-axis: "valley visits per episode".
   - One line per agent.

   - Interpretation:
       - Phase 1: the valley is attractive; high visit counts are expected.
       - Phase 2: the valley becomes neutral or mildly negative; a robust agent
         should reduce its visit rate as it discovers the corrected signal.
       - Phase 3: noisy TD-updates are injected into some LevoThinking heads;
         a resilient ensemble reduces its dependence on the corrupted region,
         keeping valley visits under control despite internal noise.