# CHARM — Enchanted Valley

Este módulo implementa el quark **CHARM** sobre el entorno Enchanted Valley.

- El entorno es un grafo no estacionario con varias rutas y fases de riesgo.
- `CharmField` construye un campo estructural a partir del subgrafo explorado.
- Un agente Q-learning tabular usa este campo como *shaping* potencial, con un
  acoplamiento λ ajustado por un optimizador tipo Lion (sign–momentum).

El archivo principal es:

- `levo/charm_enchanted_valley.py` — contiene:
    - la definición del entorno Enchanted Valley,
    - la clase `CharmField`,
    - el agente `CharmAgent`,
    - la función `train_charm_enchanted_valley(...)` que entrena al agente y
      devuelve estadísticas listas para graficar.

El notebook `notebooks/Charm_EnchantedValley_core.ipynb` muestra un flujo mínimo:

1. Ajustar unos cuantos hiperparámetros estratégicos (episodios, γ, λ inicial, etc.).
2. Ejecutar `train_charm_enchanted_valley`.
3. Visualizar recompensa, λ y ρ̄ por episodio.
