# CHARM — Modelo de campo estructural en Enchanted Valley

## 1. Rol dentro de la arquitectura Quarks

CHARM es el quark orientado a **aprovechar estructura** en entornos con
topología rica y dinámicas no estacionarias. No cambia el objetivo de
recompensa, pero deforma el paisaje de decisión mediante un campo potencial
aprendido sobre el grafo explorado.

El entorno Enchanted Valley se modela como un grafo dirigido con:

- rutas "cortas" que en ciertas fases se vuelven riesgosas,
- caminos alternativos más largos pero seguros,
- cambios de fase que modifican los riesgos de algunas aristas.

CHARM introduce un campo Φ(s) y un acoplamiento λ tal que la política efectiva
del agente tabular es:

\[
\pi(a\mid s) \propto
\exp\bigl(Q(s,a) \; + \; \lambda \cdot \Delta \Phi(s,a)\bigr)
\]

donde:

- `Q(s,a)` es el valor tabular aprendido por TD,
- `ΔΦ(s,a)` es el cambio de potencial estimado por `CharmField`,
- `λ` se ajusta con un paso de meta–control (Lion) en función de la ganancia
  estructural observada.

## 2. Señales internas

CHARM mantiene, entre otros:

- `edge_costs(s,s')` — costo estructural estimado para cada arista
  (a partir de recompensas y TD–error).
- `V(s)` y `ρ(s)` — crítico local y "polarización isomérica" derivada del
  TD–error.
- `diffusion_field(s)` — campo de difusión K–pasos desde los goals.
- `lambda_field` — intensidad actual del shaping estructural.

Estos términos se recombinan para producir campos potenciales que favorecen
trayectorias robustas frente a cambios de fase en el valle.

## 3. Uso recomendado

- Explorar cómo evoluciona la recompensa cuando se activan o no los campos
  de Charm.
- Observar la interacción entre λ y la no estacionariedad del entorno.
- Usar Enchanted Valley como *testbed* controlado para ideas de shaping
  estructural antes de pasar a entornos más grandes.
