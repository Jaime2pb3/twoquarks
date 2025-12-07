# Visualizaciones sugeridas (GRAPHICS)

Este modelo no busca gráficas bonitas, sino **gráficas honestas** que revelen:

- dónde Charm aprovecha la estructura,
- cuándo Lion corrige el acoplamiento λ,
- cómo responde el sistema a los cambios de fase del Enchanted Valley.

A continuación se sugieren algunas figuras clave.

---

## 1. Recompensa por episodio

Objetivo: ver cómo evoluciona el desempeño bruto del agente.

```python
import matplotlib.pyplot as plt
from charm_enchanted_valley import train_charm_enchanted_valley

stats = train_charm_enchanted_valley(n_episodes=300)
rewards = stats["rewards"]

plt.figure(figsize=(7,4))
plt.plot(rewards)
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.title("Recompensa por episodio en Enchanted Valley + Charm")
plt.tight_layout()
plt.show()
```

No se recomienda suavizar agresivamente; si quieres entender tendencia, puedes añadir una media móvil ligera, pero manteniendo la serie cruda visible.

---

## 2. Evolución de λ (acoplamiento al campo)

Objetivo: observar cómo **Lion ajusta la confianza en el campo estructural**.

```python
lam = stats["lambda"]

plt.figure(figsize=(7,4))
plt.plot(lam)
plt.xlabel("Episodio")
plt.ylabel("λ (Charm field coupling)")
plt.title("Evolución de λ bajo Lion")
plt.tight_layout()
plt.show()
```

Es normal observar saltos bruscos cuando el entorno cambia de fase; esos saltos son parte del comportamiento robusto.

---

## 3. Polarización isomérica media ρ̄ por episodio

Objetivo: ver si el sistema tiende a ser más **prudente** o más **ambicioso** en distintas fases.

```python
rho_mean = stats["rho_mean"]

plt.figure(figsize=(7,4))
plt.plot(rho_mean)
plt.xlabel("Episodio")
plt.ylabel("ρ medio")
plt.title("Polarización isomérica promedio por episodio")
plt.tight_layout()
plt.show()
```

- Valores cercanos a 0 → dominio de campo Safety (Φ_L).
- Valores cercanos a 1 → dominio de campo Efficiency (Φ_R).

---

## 4. Campos Φ_L, Φ_R, Φ_diff en un corte temporal

Para diagnosticar el campo, puedes modificar `charm_enchanted_valley.py` para exponer el objeto `CharmField` tras el entrenamiento y ploteas:

```python
phi_L = charm.phi_L
phi_R = charm.phi_R
phi_diff = charm.phi_diff

plt.figure(figsize=(8,4))
plt.subplot(1,3,1)
plt.bar(range(len(phi_L)), phi_L)
plt.title("Φ_L (Safety)")

plt.subplot(1,3,2)
plt.bar(range(len(phi_R)), phi_R)
plt.title("Φ_R (Efficiency)")

plt.subplot(1,3,3)
plt.bar(range(len(phi_diff)), phi_diff)
plt.title("Φ_diff (Difusión global)")

plt.tight_layout()
plt.show()
```

Esto te permite ver **cómo Charm “ve” el valle encarnado en potenciales**, y contrastarlo con el comportamiento del agente.

---

## 5. Lectura crítica

No se recomienda ocultar la irregularidad:

- si la recompensa oscila, eso indica interacción real entre estructura y no estacionariedad;
- si λ salta, es señal de que Lion está reaccionando a evidencias fuertes, no ruido suave;
- si ρ se mueve, es indicador de cambios locales en sorpresa/estabilidad.

Las mejores gráficas aquí no son las más suaves, sino las que **muestran con claridad dónde el modelo está siendo forzado a adaptarse**.
