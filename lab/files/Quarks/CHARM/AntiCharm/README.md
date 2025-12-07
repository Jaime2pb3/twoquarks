# Anti-CHARM v2

The Anti-CHARM antiquark for contextual regularization in environments with
“enchanted valleys” and reward cycles with low progress.

This module is part of the TwoQuarks family:

- **CHARM** → seeks efficient trajectories toward the goal.
- **ANTI-CHARM** → monitors loops, deceptive valleys, and diversity collapse.

Anti-CHARM does not replace the main actor; it reshapes the decision landscape
via a risk penalty:

Q_eff(s,a) = Q_charm(s,a) - lambda_t * P_anti(s,a)

where P_anti(s,a) combines:

structural loop risk,

attraction to reward valleys without progress,

contextual pressure (concentrated reward + visitation),

and lambda_t adjusts the relative weight of Anti-CHARM based on contextual
temperature (reward density and diversity).

Files

anti_charm_agent.py — Anti-CHARM v2 implementation.

MODEL.md — mathematical and design description.

README.md — this file.

Basic Usage:
from anti_charm_agent import AntiCharmAgent, AntiCharmConfig

cfg = AntiCharmConfig(
    num_states=env.num_states,
    num_actions=env.num_actions,
    device="cpu",
)
anti = AntiCharmAgent(cfg)

for episode in range(num_episodes):
    s = env.reset()
    done = False
    step = 0
    reward_history = []

    while not done:
        # 1) Q-values from CHARM
        Q_charm = charm.q_values(s)

        # 2) context stats (from actor/environment)
        stats = {
            "H_policy": float(charm.policy_entropy(s)),
            "temp": float(charm.temperature),
            "diversity": float(charm.current_diversity),
            "reward_density": float(sum(reward_history) / (len(reward_history) + 1e-6)),
            "step_norm": step / max(env.max_steps - 1, 1),
        }

        # 3) Anti-CHARM penalties
        P_anti, lambda_t = anti.penalty_vector(s, stats)
        Q_eff = Q_charm - lambda_t * P_anti

        a = int(Q_eff.argmax())
        s_next, r, done, info = env.step(a)
        reward_history.append(r)

        anti.observe_step(
            s=s,
            a=a,
            r=r,
            s_next=s_next,
            progress=float(info.get("progress", 0.0)),
            H_policy=stats["H_policy"],
            temp=stats["temp"],
            diversity=stats["diversity"],
            episode_id=episode,
            step=step,
            done=done,
        )

        s = s_next
        step += 1

    anti.end_episode()

Assumptions

The environment has a bounded discrete state space (e.g., a 7x7 grid).

There exists a meaningful progress metric toward the goal.

Policy entropy and diversity statistics can be estimated.

If the number of states exceeds max_fw_states, Anti-CHARM safely disables
Floyd–Warshall and relies solely on local statistics.

What It Mitigates

Loops in high-reward regions that make no progress toward the goal.

Contextual “climax bias” where local narrative dominates the policy.

Diversity collapse in agent trajectories.

Rather than detecting malicious prompts, Anti-CHARM measures the dynamic effect
of context: where visits and rewards concentrate and how that pressure
deforms the policy.

Limitations

Anti-CHARM is not an absolute safety mechanism.

Performance depends on reasonably defined progress and diversity signals.

Floyd–Warshall is suitable only for environments with moderate state counts.

Even so, it provides a solid and extensible baseline for experimenting with
antiquark-style regulators aimed at mitigating enchanted valleys and
contextual over-optimization.

