# Anti-CHARM v2

Antiquark de CHARM para regularización contextual en entornos con
"valles encantados" y ciclos de alto reward con bajo progreso.

Este módulo está pensado como parte de la familia TwoQuarks:

- **CHARM**  → busca trayectorias eficientes hacia el objetivo.
- **ANTI-CHARM** → vigila loops, valles engañosos y colapso de diversidad.

Anti-CHARM no reemplaza al actor principal; deforma el paisaje de decisión
mediante una penalización de riesgo:

```text
Q_eff(s,a) = Q_charm(s,a) - lambda_t * P_anti(s,a)
```

donde `P_anti(s,a)` combina:
- riesgo de loop estructural,
- atracción a valles de reward sin progreso,
- presión contextual (recompensa + visitas concentradas).

y `lambda_t` ajusta cuánto peso tiene Anti-CHARM según la temperatura actual
del contexto (reward_density y diversidad).

## Archivos

- `anti_charm_agent.py` — implementación del agente Anti-CHARM v2.
- `MODEL.md` — descripción matemática y de diseño del modelo.
- `README.md` — este archivo.

## Uso básico

```python
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
        # 1) Q-values del actor CHARM
        Q_charm = charm.q_values(s)  # shape: (num_actions,)

        # 2) stats de contexto (ej. desde el propio actor/entorno)
        stats = {
            "H_policy": float(charm.policy_entropy(s)),
            "temp": float(charm.temperature),
            "diversity": float(charm.current_diversity),
            "reward_density": float(sum(reward_history) / (len(reward_history) + 1e-6)),
            "step_norm": step / max(env.max_steps - 1, 1),
        }

        # 3) penalizaciones Anti-CHARM
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
```

### Suposiciones

- El entorno tiene **espacio de estados discreto y acotado** (p.ej. grid 7x7).
- Existe una métrica de **progreso** hacia el objetivo (aunque sea heurística).
- Se pueden estimar estadísticas como entropía de política y diversidad.

Si el número de estados supera `max_fw_states`, Anti-CHARM desactiva de forma
segura el uso de Floyd–Warshall y se apoya únicamente en estadísticas locales.

## Qué problemas busca mitigar

- Loops en regiones de alto reward que no llevan al objetivo.
- Sesgos de "clímax contextual" donde la narrativa local domina la política.
- Colapso de diversidad en las trayectorias (pérdida de alternativas sanas).

En lugar de intentar detectar prompts "maliciosos", Anti-CHARM mide el efecto
dinámico del contexto: **dónde se concentran las visitas y el reward** y cómo
se deforma la política bajo esa presión.

## Limitaciones

- No es un mecanismo de seguridad absoluto.
- Su desempeño depende de que las señales de progreso y diversidad estén
  razonablemente bien definidas.
- El uso de Floyd–Warshall es adecuado sólo para entornos con número moderado
  de estados efectivos.

Aun así, ofrece una base sólida y extensible para experimentar con antiquarks
orientados a **mitigar valles encantados y sobre-optimización contextual**.
