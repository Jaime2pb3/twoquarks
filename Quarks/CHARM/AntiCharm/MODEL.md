# Anti-CHARM v2 — Contextual Risk Regularizer

## 1. Rol dentro de la arquitectura Quarks

Anti-CHARM es el antiquark del módulo CHARM. No busca la recompensa directa,
sino **limitar el efecto de valles encantados y ciclos contextuales** que pueden
sobrepasar las políticas de alineamiento de un agente principal.

La decisión efectiva se realiza con:

\[
Q_{\text{eff}}(s,a) = Q_{\text{charm}}(s,a) - \lambda_t \cdot P_{\text{anti}}(s,a)
\]

donde:

- `Q_charm(s,a)`: valor propuesto por el actor “enamorado” (ruta más corta, reward).
- `P_anti(s,a)`: penalización de riesgo contextual estimada por Anti-CHARM.
- `\lambda_t`: ganancia isomérica dinámica, en el rango `[lambda_min, lambda_max]`.

Cuando el contexto se "calienta" (recompensas concentradas, baja diversidad),
`\lambda_t` aumenta y Anti-CHARM tiene más peso. En entornos fríos y variados,
`\lambda_t` baja y CHARM domina.

## 2. Componentes de riesgo

Anti-CHARM descompone el riesgo en tres términos analíticos:

1. **Riesgo de ciclo (`loop_risk`)**

   Se construye un grafo de estados usando las transiciones observadas en el entorno.
   Sobre ese grafo se ejecuta Floyd–Warshall para estimar el coste mínimo de volver
   al mismo estado `s` tras recorrer cualquier camino:

   \[
   loop\_cost(s) = dist(s,s)
   \]

   Este coste se normaliza con una `tanh` para evitar explosiones:

   \[
   loop\_risk(s) = \tanh\left( \frac{loop\_cost(s)}{10} \right)
   \]

   Si en el grafo se observan ciclos cortos con coste bajo, el riesgo aumenta.

2. **Puntaje de valle (`valley_score`)**

   Para cada arista agregada `(s, a, s')` se lleva una estadística de:

   - recompensa media `\bar r(s,a)`,
   - progreso medio hacia la meta `\bar p(s,a)`.

   El valle encantado se caracteriza por **recompresas altas con poco avance**:

   \[
   valley\_{raw}(s,a) = \max(0, \bar r(s,a) - \bar p(s,a))
   \]

   Se normaliza también con `tanh`:

   \[
   valley\_score(s,a) = \tanh(valley\_{raw}(s,a))
   \]

3. **Presión contextual (`context_pressure`)**

   Mide cuánto reward y cuántas visitas se concentran en la vecindad de `s`:

   \[
   context\_pressure(s) = \tanh\Bigg(
       \frac{\sum_{a,s'} R(s,a,s')}{\sum_{a,s'} N(s,a,s') + \epsilon}
   \Bigg)
   \]

La penalización base se define como combinación lineal:

\[
P_{\text{base}}(s,a) =
    \alpha_{loop}\,loop\_risk(s) +
    \beta_{valley}\,valley\_score(s,a) +
    \gamma_{context}\,context\_pressure(s)
\]

con pesos `alpha_loop`, `beta_valley` y `gamma_context` configurables.

## 3. Calibrador de riesgo aprendido

La penalización base captura estructura global, pero puede ser rígida. Para
adaptarse al entorno real, Anti-CHARM incluye un pequeño **calibrador de riesgo**
`RiskCalibrator`, entrenado mediante regresión supervisada.

Entrada del calibrador por paso:

- `p_base`          – penalización base analítica.
- `step_norm`       – paso normalizado dentro del episodio.
- `H_policy`        – entropía de la política (exploración / determinismo).
- `temp`            – temperatura efectiva del actor.
- `diversity`       – diversidad interna de acciones / trayectorias.
- `mean_reward_ep`  – recompensa media del episodio.

Salida:

- `delta_penalty` – ajuste escalar que corrige la penalización base.

Penalización final:

\[
P_{\text{anti}}(s,a) = P_{\text{base}}(s,a) + \Delta P_{\theta}(s,a)
\]

donde `\Delta P_\theta` es la salida del calibrador.

### 3.1. Target explícito de riesgo

En cada episodio se definen métricas globales:

- **Diversidad del episodio**  
  \(diversity\_ep = \frac{|\{s_t\}|}{T}\).

- **loop_rate**  
  \(loop\_rate = 1 - diversity\_ep\).

- **diversity_gap**  
  diferencia positiva respecto a una diversidad objetivo `diversity_target`.

- **reward_collapse**  
  \(reward\_collapse = \frac{\sigma(r_t)}{|\bar r_t| + \epsilon}\).

El **target de riesgo** del episodio es:

\[
risk\_label = \tanh(loop\_rate + diversity\_gap + reward\_collapse)
\]

Ese valor se asigna como etiqueta `y` para todos los pasos del episodio y el
calibrador se entrena por MSE:

\[
\mathcal{L} = \mathbb{E}[(\Delta P_\theta - risk\_label)^2]
\]

Con esto, el calibrador aprende a aumentar la penalización cuando el episodio
exhibe loops, baja diversidad o colapso de reward, sin depender de heurísticas
opacas.

## 4. Ganancia isomérica analítica \(\lambda_t\)

Para evitar una ganancia arbitraria, `\lambda_t` se calcula de forma analítica
usando sólo dos señales globales:

- densidad de recompensa `reward_density`,
- diversidad actual `diversity`.

Primero se construye un score en aproximadamente `[-1, 1]`:

\[
score = \tanh(reward\_density) + (diversity\_{target} - diversity)
\]

Este score se satura a `[-1,1]` y luego se mapea al intervalo
`[lambda_min, lambda_max]`:

\[
\lambda_t = mid + span \cdot score,
\]

donde `mid = (lambda_min + lambda_max)/2` y `span` es la mitad del rango.

Resultado:

- Contextos calientes y poco diversos ⟹ `\lambda_t` se acerca a `lambda_max`.
- Contextos fríos y diversos      ⟹ `\lambda_t` se acerca a `lambda_min`.

No hay entrenamiento oculto ni meta-aprendizaje inestable; el comportamiento
es interpretable y controlable por diseño.

## 5. Puntos débiles y mitigaciones

1. **Complejidad de Floyd–Warshall O(n³)**  
   - Mitigación: se limita el número máximo de estados (`max_fw_states`) y sólo
     se ejecuta cada cierto número de episodios (`fw_interval_episodes`). El uso
     recomendado es en entornos discretos pequeños (p.ej. hasta unos cientos de
     estados efectivos).

2. **Estados sin datos suficientes**  
   - Mitigación: para `(s,a)` sin historial, se usa una penalización suave y se
     mantiene `context_pressure` cercano a cero. Esto evita castigos
     desproporcionados por simple falta de observación.

3. **Sobrepenalización (agente demasiado paranoico)**  
   - Mitigación: las contribuciones de riesgo se pasan por `tanh`, de forma que
     estén acotadas. Además, los pesos `alpha_loop`, `beta_valley` y
     `gamma_context` son configurables y pueden ajustarse empíricamente.

4. **Etiquetas de riesgo demasiado gruesas**  
   - Mitigación: aunque el risk_label se calcula a nivel de episodio, cada paso
     incluye `p_base`, `step_norm`, `diversity` y el resto de features, lo que
     permite que el calibrador aprenda patrones intra-episodio sin necesidad de
     señales más complejas.

## 6. Integración con CHARM

1. El actor principal (CHARM) calcula `Q_charm(s,a)` para todas las acciones.
2. Anti-CHARM calcula `P_anti(s,a)` y `lambda_t` con `penalty_vector`.
3. La política efectiva se define sobre:

   \[
   Q_{\text{eff}}(s,a) = Q_{\text{charm}}(s,a) - \lambda_t P_{\text{anti}}(s,a)
   \]

4. El agente ejecuta `argmax_a Q_eff(s,a)` o aplica `softmax(Q_eff)` si trabaja
   con políticas estocásticas.

De esta forma, Anti-CHARM funciona como **regulador contextual**: no censura
acciones específicas, sino que reconfigura el paisaje de decisión para que
los valles encantados y los loops tengan menos atractivo efectivo.
