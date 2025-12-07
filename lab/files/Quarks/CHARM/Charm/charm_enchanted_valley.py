#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Charm • Enchanted Valley
========================

Entorno "Enchanted Valley" + módulo CharmField + agente base + loop de entrenamiento.

Idea central:
- El entorno es un grafo no estacionario con múltiples rutas.
- Charm construye campos estructurales a partir del subgrafo explorado.
- El agente base hace Q-learning tabular.
- La política efectiva se acopla al campo con un factor λ aprendido con Lion
  (sign-momentum) a nivel meta-control.

Este archivo está diseñado para ser auto contenido y didáctico, no hiper-optimizado.
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Entorno: Enchanted Valley (grafo no estacionario)
# ---------------------------------------------------------------------------

class EnchantedValleyEnv:
    """
    Grafo pequeño pero topológicamente rico:
    - 3 zonas: A (entrada), B (meseta), C (núcleo).
    - Varias rutas alternativas al núcleo.
    - Algunas aristas se vuelven riesgosas según la fase.
    - Fases cambian cada N episodios (no estacionario).

    Estados: enteros 0..N-1
    Acciones: índices sobre lista de vecinos del estado actual.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.n_states = 14  # pequeño pero no trivial
        # Grafo estático: lista de vecinos (sin pesos aún)
        self.adj = self._build_graph_structure()
        self.start_state = 0
        self.goal_states = {11, 13}  # dos nodos en "núcleo C"

        self.max_steps = 40
        self.episode_idx = 0
        self.phase = 0
        self.state = self.start_state
        self.steps = 0

    def _build_graph_structure(self) -> Dict[int, List[int]]:
        """
        Grafo manualmente diseñado:

        Zona A: 0,1,2
        Zona B: 3,4,5,6,7
        Zona C: 8,9,10,11,12,13

        Rutas:
        0 -> 1 -> 3 -> 5 -> 8 -> 11 (ruta "corta" hacia C1)
        0 -> 2 -> 4 -> 6 -> 9 -> 13 (ruta "alternativa" hacia C2)
        3 -> 7 -> 10 -> 12 -> 11 (desvío interno)
        """
        adj = defaultdict(list)
        edges = [
            (0, 1), (0, 2),
            (1, 3),
            (2, 4),
            (3, 5), (3, 7),
            (4, 6),
            (5, 8),
            (6, 9),
            (7, 10),
            (8, 11),
            (9, 13),
            (10, 12),
            (12, 11),
            # algunas conexiones laterales
            (4, 3),
            (6, 5),
            (9, 10),
        ]
        for u, v in edges:
            adj[u].append(v)
        return adj

    def _update_phase(self):
        """
        Fases del entorno (no estacionario):

        phase 0: todas las aristas tienen costo similar.
        phase 1: ciertas aristas cortas se vuelven riesgosas.
        phase 2: se "abren" rutas alternativas (modificando recompensas).
        """
        # Ejemplo simple: fase por ventana de episodios
        if self.episode_idx < 100:
            self.phase = 0
        elif self.episode_idx < 200:
            self.phase = 1
        else:
            self.phase = 2

    def reset(self) -> int:
        self._update_phase()
        self.state = self.start_state
        self.steps = 0
        return self.state

    def get_neighbors(self, s: int) -> List[int]:
        return self.adj[s]

    def step(self, action_idx: int) -> Tuple[int, float, bool, Dict]:
        """
        action_idx: índice dentro de self.adj[state].
        """
        self.steps += 1
        neighbors = self.adj[self.state]
        if not neighbors:
            # estado sin acciones -> episodio termina mal
            return self.state, -5.0, True, {"terminal_reason": "dead_end"}

        action_idx = max(0, min(action_idx, len(neighbors) - 1))
        next_state = neighbors[action_idx]

        # Recompensa base: costo por paso
        reward = -1.0

        # Bonus por goal
        done = False
        info = {}

        if next_state in self.goal_states:
            reward += 15.0
            done = True
            info["terminal_reason"] = "goal"

        # Penalizaciones / riesgos según fase y aristas
        u, v = self.state, next_state

        # Fase 1: camino "corto" 0-1-3-5-8-11 se vuelve riesgoso en 3->5 y 5->8
        if self.phase == 1:
            risky_edges = {(3, 5), (5, 8)}
            if (u, v) in risky_edges:
                # 30% de chance de penalización fuerte
                if self.rng.random() < 0.3:
                    reward -= 10.0
                    info["risk_event"] = True

        # Fase 2: algunos edges "laterales" se vuelven menos costosos
        if self.phase == 2:
            alt_edges = {(4, 6), (6, 9), (9, 10)}
            if (u, v) in alt_edges:
                reward += 0.5  # incentivo suave

        # Terminar por límite de pasos
        if self.steps >= self.max_steps and not done:
            done = True
            info["terminal_reason"] = "max_steps"

        self.state = next_state
        return next_state, reward, done, info


# ---------------------------------------------------------------------------
# 2. CharmField Engine (campos + isomería + Lion meta-control)
# ---------------------------------------------------------------------------

@dataclass
class CharmConfig:
    gamma: float = 0.95

    # critic para TD-error y rho
    alpha_v: float = 0.1

    # actualización de costo de aristas
    alpha_cost: float = 0.2

    # meta-control Lion sobre lambda
    lr_lambda: float = 0.01
    beta1_lambda: float = 0.9

    # parámetros campo
    lambda_init: float = 0.5
    mu_diff: float = 0.2  # peso fijo sobre campo difuso global

    # difusión topológica
    diffusion_K: int = 4
    diffusion_gamma: float = 0.8

    # polarización isomérica
    kappa_rho: float = 2.0       # sensibilidad a |TD-error|
    rho_min: float = 0.05
    rho_max: float = 0.95


class CharmField:
    """
    Charm: módulo de campo estructural.

    - Mantiene un subgrafo causal mediante costos de aristas observadas.
    - Calcula dos campos:
        * Safety (Φ_L): penaliza riesgo / sorpresa.
        * Efficiency (Φ_R): prioriza proximidad al goal.
    - Construye un campo difuso global por K pasos de difusión.
    - Calcula un potencial combinado Φ(s).
    - Acopla la política base mediante un factor λ (aprendido con Lion).
    - ρ(s) se deriva de |TD-error(s)| (isomeric polarization).
    """

    def __init__(self, n_states: int, goal_states, config: CharmConfig):
        self.n_states = n_states
        self.goal_states = list(goal_states)
        self.cfg = config

        # Costos empíricos (subgrafo causal)
        self.edge_counts = np.zeros((n_states, n_states), dtype=np.float32)
        self.edge_costs = np.full((n_states, n_states), np.inf, dtype=np.float32)

        # Campos
        self.phi_L = np.zeros(n_states, dtype=np.float32)   # safety
        self.phi_R = np.zeros(n_states, dtype=np.float32)   # efficiency
        self.phi_diff = np.zeros(n_states, dtype=np.float32)

        # Parámetros meta-control
        self.lambda_field = self.cfg.lambda_init
        self.mu_diff = self.cfg.mu_diff
        self.m_lambda = 0.0  # momento Lion sobre lambda

        # Critic para TD-error y rho
        self.V = np.zeros(n_states, dtype=np.float32)
        self.last_td_error = np.zeros(n_states, dtype=np.float32)

        # rho(s): polarización isomérica
        self.rho = np.zeros(n_states, dtype=np.float32)

    # --------------------------- Actualizaciones online -------------------- #

    def update_edge_cost(self, s: int, s_next: int, reward: float):
        """
        Actualiza costo de arista c(s,s') a partir de r (subgrafo causal).
        """
        self.edge_counts[s, s_next] += 1.0
        prev = self.edge_costs[s, s_next]
        if not np.isfinite(prev):
            prev = 0.0
        new_obs_cost = -reward  # costo ~ -r (puede refinarse)
        self.edge_costs[s, s_next] = (
            (1 - self.cfg.alpha_cost) * prev
            + self.cfg.alpha_cost * new_obs_cost
        )

    def update_critic_and_rho(self, s: int, r: float, s_next: int, done: bool):
        """
        TD(0) para V(s) y cálculo de TD-error => polarización isomérica ρ(s).
        """
        gamma = self.cfg.gamma
        v_s = self.V[s]
        v_next = 0.0 if done else self.V[s_next]
        td_error = r + gamma * v_next - v_s

        # actualizar crítico
        self.V[s] += self.cfg.alpha_v * td_error
        self.last_td_error[s] = td_error

        # polarización isomérica: ρ alto si sorpresa alta (exploración agresiva)
        # usamos |TD-error| para medir sorpresa
        x = self.cfg.kappa_rho * abs(td_error)
        rho_s = 1.0 / (1.0 + math.exp(-x))  # sigmoide
        # restringir rango
        rho_s = max(self.cfg.rho_min, min(self.cfg.rho_max, rho_s))
        self.rho[s] = rho_s

    # --------------------------- Cálculo de campos ------------------------- #

    def _dijkstra(self, cost_matrix: np.ndarray) -> np.ndarray:
        """
        Dijkstra multi-goal: calculamos distancias mínimas desde cada estado
        a cualquiera de los goal_states, sobre el subgrafo observado.

        cost_matrix: shape (n_states, n_states) con costos >=0 o inf.
        Devuelve array dist[s].
        """
        n = self.n_states
        dist = np.full(n, np.inf, dtype=np.float32)
        visited = np.zeros(n, dtype=bool)

        # inicializar goals con 0
        for g in self.goal_states:
            dist[g] = 0.0

        for _ in range(n):
            # seleccionar nodo no visitado con menor dist
            u = -1
            best = np.inf
            for i in range(n):
                if not visited[i] and dist[i] < best:
                    best = dist[i]
                    u = i
            if u == -1 or not np.isfinite(best):
                break
            visited[u] = True
            # relajar vecinos a partir de costos observados
            for v in range(n):
                c_uv = cost_matrix[u, v]
                if not np.isfinite(c_uv):
                    continue
                nd = best + c_uv
                if nd < dist[v]:
                    dist[v] = nd
        return dist

    def _build_safety_costs(self) -> np.ndarray:
        """
        Costos para campo Safety (L):
        - Penaliza aristas con costo alto.
        - Penaliza transiciones raras (pocas observaciones).
        """
        n = self.n_states
        cost_L = np.full((n, n), np.inf, dtype=np.float32)

        max_cost = np.nanmax(
            np.where(np.isfinite(self.edge_costs), self.edge_costs, 0.0)
        ) + 1e-6
        max_count = np.nanmax(self.edge_counts) + 1e-6

        for s in range(n):
            for sp in range(n):
                if not np.isfinite(self.edge_costs[s, sp]):
                    continue
                base = self.edge_costs[s, sp] / max_cost  # [0,1] relativo
                # penalizar aristas muy poco visitadas (consideradas "riesgosas")
                rarity = 1.0 - (self.edge_counts[s, sp] / max_count)
                cost_L[s, sp] = base + 0.5 * rarity
        return cost_L

    def _build_efficiency_costs(self) -> np.ndarray:
        """
        Costos para campo Efficiency (R):
        - Prioriza aristas con menor costo medido.
        - No penaliza rareza: fomenta rutas nuevas si parecen baratas.
        """
        n = self.n_states
        cost_R = np.full((n, n), np.inf, dtype=np.float32)

        max_cost = np.nanmax(
            np.where(np.isfinite(self.edge_costs), self.edge_costs, 0.0)
        ) + 1e-6

        for s in range(n):
            for sp in range(n):
                if not np.isfinite(self.edge_costs[s, sp]):
                    continue
                base = self.edge_costs[s, sp] / max_cost
                cost_R[s, sp] = base
        return cost_R

    def _compute_diffusion_field(self):
        """
        Campo difuso global tipo K-pasos de difusión desde goals.

        Inicializamos un vector r_goal con 1 en goals, 0 en demás;
        propagamos K veces con matriz de probabilidad P empírica.
        """
        n = self.n_states
        # construir P empírico (normalizado por fila)
        P = np.zeros((n, n), dtype=np.float32)
        for s in range(n):
            # cuenta total de salidas observadas
            counts = self.edge_counts[s]
            total = np.sum(counts)
            if total <= 0:
                continue
            for sp in range(n):
                if counts[sp] > 0:
                    P[s, sp] = counts[sp] / total

        # r_goal: 1 en goals, 0 en demás
        r = np.zeros(n, dtype=np.float32)
        for g in self.goal_states:
            r[g] = 1.0

        phi = np.zeros(n, dtype=np.float32)
        x = r.copy()
        gamma = self.cfg.diffusion_gamma

        for _ in range(self.cfg.diffusion_K):
            phi += x
            x = gamma * P.T.dot(x)

        # normalizar suavemente
        if np.max(phi) > 0:
            phi = phi / (np.max(phi) + 1e-6)
        self.phi_diff = phi

    def recompute_fields(self):
        """
        Recalcula Φ_L, Φ_R y Φ_diff sobre el subgrafo actual.
        """
        # Safety & efficiency via Dijkstra
        cost_L = self._build_safety_costs()
        cost_R = self._build_efficiency_costs()

        dist_L = self._dijkstra(cost_L)
        dist_R = self._dijkstra(cost_R)

        # convertir a potenciales (negativa de distancia)
        max_L = np.nanmax(
            np.where(np.isfinite(dist_L), dist_L, 0.0)
        ) + 1e-6
        max_R = np.nanmax(
            np.where(np.isfinite(dist_R), dist_R, 0.0)
        ) + 1e-6

        # invertir y normalizar
        phi_L = np.zeros_like(dist_L)
        phi_R = np.zeros_like(dist_R)

        for i in range(self.n_states):
            if np.isfinite(dist_L[i]):
                phi_L[i] = 1.0 - dist_L[i] / max_L
            if np.isfinite(dist_R[i]):
                phi_R[i] = 1.0 - dist_R[i] / max_R

        self.phi_L = phi_L
        self.phi_R = phi_R

        # Difusión global
        self._compute_diffusion_field()

    # ------------------- Potencial combinado + shaping --------------------- #

    def combined_potential(self, s: int) -> float:
        """
        Φ(s) = (1 - ρ) * Φ_L + ρ * Φ_R + μ * Φ_diff
        """
        rho_s = self.rho[s]
        phi_s = (1.0 - rho_s) * self.phi_L[s] + rho_s * self.phi_R[s]
        phi_s += self.mu_diff * self.phi_diff[s]
        return phi_s

    def delta_phi(self, s: int, s_next: int) -> float:
        return self.combined_potential(s_next) - self.combined_potential(s)

    # -------------------------- Lion sobre lambda -------------------------- #

    def lion_meta_update_lambda(self, avg_struct_gain: float):
        """
        Lion sobre λ usando sign-momentum.

        Intuición:
        - Si aprovechar el campo (ganancia estructural positiva) fue bueno,
          queremos aumentar λ (más shaping).
        - Si el uso del campo no aporta o empeora, bajamos λ.
        """
        # Gradiente aproximado: g ≈ -avg_struct_gain
        # (si struct_gain > 0 queremos más λ -> grad < 0)
        g = -avg_struct_gain

        b1 = self.cfg.beta1_lambda
        self.m_lambda = b1 * self.m_lambda + (1.0 - b1) * g

        step = self.cfg.lr_lambda * np.sign(self.m_lambda)
        self.lambda_field -= step  # grad descent
        # mantener λ en [0, 2] por seguridad
        self.lambda_field = max(0.0, min(2.0, self.lambda_field))


# ---------------------------------------------------------------------------
# 3. Agente base (Q-learning tabular + Charm shaping)
# ---------------------------------------------------------------------------

class CharmAgent:
    """
    Agente tabular Q-learning con shaping de Charm:

    - Q(s,a) se actualiza por TD.
    - Política efectiva:
        π(a|s) ∝ exp( Q(s,a) + λ * ΔΦ(s,a) )
    - CharmField proporciona ΔΦ(s,a) y λ se adapta por Lion.
    """

    def __init__(
        self,
        n_states: int,
        max_actions: int,
        charm: CharmField,
        gamma: float = 0.95,
        alpha_q: float = 0.2,
        temperature: float = 1.0,
    ):
        self.n_states = n_states
        self.max_actions = max_actions
        self.gamma = gamma
        self.alpha_q = alpha_q
        self.temperature = temperature
        self.charm = charm

        # Q(s,a) para un máximo de acciones; las acciones no válidas se ignoran
        self.Q = np.zeros((n_states, max_actions), dtype=np.float32)

    def select_action(self, s: int, neighbors: List[int]) -> int:
        """
        Selección de acción softmax sobre Q + λ*ΔΦ.
        neighbors: lista de estados vecinos desde s; acciones = indices 0..len(neighbors)-1
        """
        if not neighbors:
            return 0
        n_actions = len(neighbors)

        logits = np.zeros(n_actions, dtype=np.float32)
        for a in range(n_actions):
            sp = neighbors[a]
            base_q = self.Q[s, a]
            dphi = self.charm.delta_phi(s, sp)
            logits[a] = (base_q + self.charm.lambda_field * dphi) / max(
                1e-3, self.temperature
            )

        # softmax estable
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)

        # muestreo
        r = random.random()
        cdf = 0.0
        for a in range(n_actions):
            cdf += probs[a]
            if r <= cdf:
                return a
        return n_actions - 1

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        """
        Actualiza Q(s,a) con TD(0).
        """
        q_sa = self.Q[s, a]
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s_next])
        td = target - q_sa
        self.Q[s, a] += self.alpha_q * td
        return td


# ---------------------------------------------------------------------------
# 4. Loop de entrenamiento
# ---------------------------------------------------------------------------

def train_charm_enchanted_valley(
    n_episodes: int = 300,
    seed: int = 0,
    recompute_fields_every: int = 10,
):
    env = EnchantedValleyEnv(seed=seed)
    cfg = CharmConfig()
    charm = CharmField(n_states=env.n_states, goal_states=list(env.goal_states), config=cfg)

    max_actions = max(len(env.get_neighbors(s)) for s in range(env.n_states))
    agent = CharmAgent(
        n_states=env.n_states, max_actions=max_actions, charm=charm
    )

    rewards_hist = []
    lambda_hist = []
    rho_mean_hist = []

    for ep in range(n_episodes):
        env.episode_idx = ep
        s = env.reset()
        done = False
        ep_reward = 0.0
        struct_gain_sum = 0.0
        steps = 0

        while not done:
            neighbors = env.get_neighbors(s)
            a = agent.select_action(s, neighbors)
            s_next, r, done, info = env.step(a)

            # actualizar Charm: costos + crítico + rho
            charm.update_edge_cost(s, s_next, r)
            charm.update_critic_and_rho(s, r, s_next, done)

            # Q-learning base
            td = agent.update(s, a, r, s_next, done)

            # ganancia estructural observada
            dphi = charm.delta_phi(s, s_next)
            struct_gain_sum += dphi

            ep_reward += r
            steps += 1
            s = s_next

        # meta-update de lambda con Lion
        avg_struct_gain = struct_gain_sum / max(1, steps)
        charm.lion_meta_update_lambda(avg_struct_gain)

        # actualizar campos periódicamente
        if (ep + 1) % recompute_fields_every == 0:
            charm.recompute_fields()

        rewards_hist.append(ep_reward)
        lambda_hist.append(charm.lambda_field)
        rho_mean_hist.append(float(np.mean(charm.rho)))

        if (ep + 1) % 50 == 0:
            print(
                f"[Ep {ep+1:4d}] "
                f"Reward={ep_reward:6.2f}  "
                f"λ={charm.lambda_field:.3f}  "
                f"ρ_mean={rho_mean_hist[-1]:.3f}  "
                f"phase={env.phase}"
            )

    return {
        "rewards": np.array(rewards_hist),
        "lambda": np.array(lambda_hist),
        "rho_mean": np.array(rho_mean_hist),
    }


if __name__ == "__main__":
    stats = train_charm_enchanted_valley()
