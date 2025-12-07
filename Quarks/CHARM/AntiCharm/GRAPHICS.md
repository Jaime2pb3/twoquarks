# Anti-CHARM — Visualizaciones sugeridas

Anti-CHARM no busca maximizar reward directo sino moderar el comportamiento
del actor principal (CHARM u otro agente) cuando el contexto se vuelve
peligroso: loops, valles encantados y colapso de diversidad.

Algunas gráficas útiles para interpretar su efecto:

## 1. Lambda de riesgo λ_t

Trazar λ_t por episodio o por bloque de pasos para ver:

- cuándo el sistema percibe el contexto "caliente" (reward denso, baja
  diversidad),
- cuándo se relaja porque la exploración es más variada.

## 2. Penalización promedio P_anti

Para cada episodio, calcular la media de \(P_{\text{anti}}(s,a)\) sobre
las acciones visitadas. Esto indica qué tan fuerte está empujando
Anti-CHARM contra loops y valles falsos.

## 3. Reward bruto vs reward regulado

Comparar:

- recompensa acumulada del actor base con
- recompensa acumulada cuando se usa \(Q_{\text{eff}} = Q_{\text{charm}} -
  \lambda_t P_{\text{anti}}\).

El objetivo no es siempre ganar más reward, sino evitar trayectorias
patológicas (ciclos infinitos, atrapamiento en subregiones de alto reward
pero bajo progreso).

## 4. Diversidad y densidad de visitas

- Histograma de estados visitados antes y después de activar Anti-CHARM.
- Evolución del número de estados únicos visitados por episodio.

Estos gráficos ayudan a ver si Anti-CHARM está dispersando el comportamiento
o solo recortando unas pocas rutas malas.

En conjunto, estas visualizaciones permiten leer a Anti-CHARM como un
regulador contextual y no como un simple "castigo adicional" fijo.
