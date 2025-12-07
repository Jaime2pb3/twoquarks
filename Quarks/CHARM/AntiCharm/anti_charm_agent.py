"""
Anti-Charm Agent v2
--------------------
Robust contextual regularizer for CHARM-like agents.

Idea:
    Q_eff(s,a) = Q_charm(s,a) - lambda_t * P_anti(s,a)

where P_anti is a risk penalty that estimates:
    - loopiness (getting stuck in local cycles),
    - fake-valley attraction (high reward / low progress),
    - contextual pressure (reward + repetition concentrated en una zona),
  and lambda_t is a bounded gain that grows when the context "se calienta"
  (reward concentrado, baja diversidad) y baja cuando el sistema está frío.

Supone:
    - Espacio de estados discreto y relativamente pequeño (p.ej. gridworlds).
    - Progreso escalar en [0,1] (0 = lejos de la meta, 1 = meta).
    - Stats básicos por paso: entropía de la política, temperatura, diversidad.

Este módulo NO decide la acción óptima por sí solo.
Deforma el paisaje de decisión para que el actor principal (CHARM)
sea menos vulnerable a valles encantados y loops contextuales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW


@dataclass
class AntiCharmConfig:
    num_states: int
    num_actions: int

    # Pesos analíticos de la penalización base
    alpha_loop: float = 1.0      # peso del riesgo de loop
    beta_valley: float = 1.0     # peso de valles falsos
    gamma_context: float = 0.5   # presión contextual

    # Objetivos globales
    diversity_target: float = 0.6
    max_fw_states: int = 512          # límite duro para Floyd–Warshall
    fw_interval_episodes: int = 5     # cada cuántos episodios refrescar el grafo

    # Entrenamiento del calibrador de riesgo
    calib_lr: float = 1e-3
    weight_decay: float = 1e-3

    # Rango de lambda (ganancia isomérica)
    lambda_min: float = 0.05
    lambda_max: float = 0.95

    device: str = "cpu"


class RiskCalibrator(nn.Module):
    """
    Pequeña red que ajusta la penalización base usando un target de riesgo explícito.
    Entrada:  feature vector (6,)
    Salida :  delta_penalty escalar
    """
    def __init__(self, in_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim)
        return self.net(x).squeeze(-1)  # (batch,)


class AntiCharmAgent:
    """
    Implementa el antiquark CHARM (ANTI-CHARM) como módulo de riesgo contextual.

    Uso esperado (pseudocódigo):

        anti = AntiCharmAgent(cfg)

        for episode in range(N):
            s = env.reset()
            done = False
            step = 0
            while not done:
                # 1) actor principal propone valores
                Q_charm = charm.q_values(s)      # shape: (num_actions,)

                # 2) Anti-Charm calcula penalizaciones por acción
                stats = {...}  # H_policy, temp, diversity, reward_density, ...
                P_anti, lambda_t = anti.penalty_vector(s, stats)  # shape: (num_actions,)

                Q_eff = Q_charm - lambda_t * P_anti
                a = int(Q_eff.argmax())

                s_next, r, done, info = env.step(a)

                # 3) registrar transición para Anti-Charm
                anti.observe_step(
                    s=s, a=a, r=r, s_next=s_next,
                    progress=info.get("progress", 0.0),
                    H_policy=stats.get("H_policy", 0.0),
                    temp=stats.get("temp", 1.0),
                    diversity=stats.get("diversity", 0.0),
                    episode_id=episode,
                    step=step,
                    done=done,
                )

                s = s_next
                step += 1

            anti.end_episode()

    """

    def __init__(self, cfg: AntiCharmConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        n_s = cfg.num_states
        n_a = cfg.num_actions

        # Estadísticas de transiciones por arista (s,a,s')
        self.edge_counts = np.zeros((n_s, n_a, n_s), dtype=np.int32)
        self.edge_reward = np.zeros((n_s, n_a, n_s), dtype=np.float32)
        self.edge_progress = np.zeros((n_s, n_a, n_s), dtype=np.float32)
        self.edge_steps = np.zeros((n_s, n_a, n_s), dtype=np.float32)

        # Distancias para Floyd–Warshall (se calculan sobre un grafo colapsado s -> s')
        self.dist = np.full((n_s, n_s), np.inf, dtype=np.float32)
        np.fill_diagonal(self.dist, 0.0)

        # Buffers de episodio
        self.current_episode_steps: List[Dict[str, Any]] = []
        self.episode_counter: int = 0

        # Calibrador de riesgo
        self.calibrator = RiskCalibrator(in_dim=6).to(self.device)
        self.opt_calib = AdamW(
            self.calibrator.parameters(),
            lr=cfg.calib_lr,
            weight_decay=cfg.weight_decay,
        )

    # ------------------------------------------------------------------
    #  MONITOREO ONLINE
    # ------------------------------------------------------------------
    def observe_step(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        progress: float,
        H_policy: float,
        temp: float,
        diversity: float,
        episode_id: int,
        step: int,
        done: bool,
    ) -> None:
        """
        Registrar una transición (s, a, r, s') junto con stats contextuales.
        """
        # actualizar stats de arista
        self._update_edge_stats(s, a, s_next, r, progress)

        # buffer para entrenamiento episodico
        self.current_episode_steps.append(
            dict(
                s=s,
                a=a,
                r=r,
                s_next=s_next,
                progress=progress,
                H_policy=H_policy,
                temp=temp,
                diversity=diversity,
                step=step,
                done=done,
            )
        )

    def _update_edge_stats(self, s: int, a: int, s_next: int, r: float, progress: float) -> None:
        """
        Acumula estadísticas para la arista (s, a, s_next).
        """
        self.edge_counts[s, a, s_next] += 1
        self.edge_reward[s, a, s_next] += float(r)
        self.edge_progress[s, a, s_next] += float(progress)
        self.edge_steps[s, a, s_next] += 1.0

    # ------------------------------------------------------------------
    #  PENALIZACIÓN ONLINE
    # ------------------------------------------------------------------
    def penalty_vector(self, s: int, stats: Dict[str, float]) -> Tuple[np.ndarray, float]:
        """
        Devuelve el vector de penalizaciones P_anti(s, a) para todas las acciones
        y el lambda_t actual. No altera pesos ni entrena nada.
        """
        n_a = self.cfg.num_actions
        p_vec = np.zeros((n_a,), dtype=np.float32)

        # calcular lambda_t a partir de stats de contexto global
        lambda_t = self._compute_lambda_analytic(stats)

        for a in range(n_a):
            p_vec[a] = self._penalty_single_action(s, a, stats)

        return p_vec, float(lambda_t)

    def _penalty_single_action(self, s: int, a: int, stats: Dict[str, float]) -> float:
        """
        Penalización total para una acción específica en un estado dado.
        Se compone de:
            P_anti = P_base + delta_calib
        """
        # base analítica
        p_base, features = self._compute_penalty_base(s, a)

        # features para el calibrador de riesgo
        feat_vec = np.array(
            [
                p_base,
                stats.get("step_norm", 0.0),
                stats.get("H_policy", 0.0),
                stats.get("temp", 1.0),
                stats.get("diversity", 0.0),
                stats.get("reward_density", 0.0),
            ],
            dtype=np.float32,
        )
        x = torch.tensor(feat_vec, device=self.device).unsqueeze(0)
        with torch.no_grad():
            delta = self.calibrator(x).item()

        return float(p_base + delta)

    # ------------------------------------------------------------------
    #  CÁLCULO DE P_base
    # ------------------------------------------------------------------
    def _refresh_distances_if_needed(self) -> None:
        """
        Corre Floyd–Warshall sobre el grafo colapsado s->s' si:
          - el número de estados es razonable
          - han pasado suficientes episodios
        """
        cfg = self.cfg
        if cfg.num_states > cfg.max_fw_states:
            # grafo demasiado grande: dist se queda como identidades / inf
            return

        # construir matriz de costos por transición (s -> s')
        n_s = cfg.num_states
        cost = np.full((n_s, n_s), np.inf, dtype=np.float32)
        np.fill_diagonal(cost, 0.0)

        for s in range(n_s):
            for sp in range(n_s):
                counts_sa = self.edge_counts[s, :, sp].sum()
                if counts_sa == 0:
                    continue
                # reward medio y progreso medio desde s hasta sp
                mean_r = self.edge_reward[s, :, sp].sum() / max(counts_sa, 1)
                mean_prog = self.edge_progress[s, :, sp].sum() / max(counts_sa, 1)

                # costo: 1 paso + penalización por reward "sospechosamente alto"
                # con poco progreso (valles encantados)
                valley_term = max(0.0, mean_r - mean_prog)
                step_cost = 1.0 + valley_term
                cost[s, sp] = min(cost[s, sp], step_cost)

        # Floyd–Warshall clásico
        dist = cost.copy()
        for k in range(n_s):
            dist = np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])

        self.dist = dist

    def _compute_penalty_base(self, s: int, a: int) -> Tuple[float, Dict[str, float]]:
        """
        Combina tres componentes:

            - loop_risk(s):  ciclos cortos s -> ... -> s con costo bajo.
            - valley_score(s,a): reward alto / progreso bajo.
            - context_pressure(s): reward + visitas acumuladas alrededor de s.

        Se normaliza con tanh para evitar explosiones.
        """
        cfg = self.cfg

        # riesgo de loop: buscamos el ciclo más barato que empieza y termina en s
        # (si no hay ciclo observado, esto será grande).
        loop_cost = float(self.dist[s, s])
        loop_risk = math.tanh(loop_cost / 10.0)  # normalización suave

        # stats de la arista específica (s,a,*)
        counts = self.edge_counts[s, a, :].sum()
        if counts == 0:
            # sin datos: penalización suave pero no infinita
            valley_score = 0.1
            context_pressure = 0.1
        else:
            mean_r = self.edge_reward[s, a, :].sum() / counts
            mean_prog = self.edge_progress[s, a, :].sum() / counts
            visits_from_s = self.edge_counts[s, :, :].sum()
            total_reward_from_s = self.edge_reward[s, :, :].sum()

            valley_raw = max(0.0, mean_r - mean_prog)
            valley_score = math.tanh(valley_raw)

            if visits_from_s > 0:
                context_pressure = math.tanh(
                    (total_reward_from_s / visits_from_s)
                )
            else:
                context_pressure = 0.0

        p_base = (
            cfg.alpha_loop * loop_risk
            + cfg.beta_valley * valley_score
            + cfg.gamma_context * context_pressure
        )

        features = dict(
            loop_risk=loop_risk,
            valley_score=valley_score,
            context_pressure=context_pressure,
        )
        return float(p_base), features

    # ------------------------------------------------------------------
    #  LAMBDA ANALÍTICO
    # ------------------------------------------------------------------
    def _compute_lambda_analytic(self, stats: Dict[str, float]) -> float:
        """
        Calcula lambda_t sin aprendizaje opaco, solo a partir de stats globales:

            - Subirá cuando haya:
                * reward_density alto,
                * diversidad baja.

            - Bajará cuando el sistema esté variado y frío.
        """
        cfg = self.cfg

        reward_density = stats.get("reward_density", 0.0)
        diversity = stats.get("diversity", 0.0)

        # normalización suave
        r_term = math.tanh(reward_density)
        d_term = (cfg.diversity_target - diversity)

        # score en [-1, 1] aprox.
        raw = r_term + d_term
        score = max(-1.0, min(1.0, raw))

        # mapear a [lambda_min, lambda_max]
        mid = (cfg.lambda_min + cfg.lambda_max) / 2.0
        span = (cfg.lambda_max - cfg.lambda_min) / 2.0
        lam = mid + span * score
        return float(max(cfg.lambda_min, min(cfg.lambda_max, lam)))

    # ------------------------------------------------------------------
    #  ENTRENAMIENTO EPISÓDICO DEL CALIBRADOR
    # ------------------------------------------------------------------
    def end_episode(self) -> None:
        """
        Debe llamarse al final de cada episodio.
        - Actualiza el grafo (cada cierto número de episodios).
        - Entrena el calibrador de riesgo con un target explícito.

        El target de riesgo del episodio mezcla:
            - loop_rate       (número de estados repetidos / pasos)
            - diversity_gap   (diversidad objetivo - diversidad real, si es positivo)
            - reward_collapse (std alto de reward relativo al rango)
        """
        if not self.current_episode_steps:
            return

        self.episode_counter += 1

        steps = self.current_episode_steps
        num_steps = len(steps)

        states = [s["s"] for s in steps]
        rewards = [s["r"] for s in steps]
        diversities = [s["diversity"] for s in steps]

        unique_states = len(set(states))
        diversity_ep = unique_states / max(num_steps, 1)

        # tasa de repetición de estados (loop_rate)
        loop_rate = 1.0 - diversity_ep

        # gap respecto al objetivo de diversidad
        diversity_gap = max(0.0, self.cfg.diversity_target - diversity_ep)

        # colapso de reward: std alta en relación a media
        mean_r = float(np.mean(rewards)) if num_steps > 0 else 0.0
        std_r = float(np.std(rewards)) if num_steps > 0 else 0.0
        reward_collapse = std_r / (abs(mean_r) + 1e-6)

        # riesgo global del episodio (bounded)
        risk_label = math.tanh(loop_rate + diversity_gap + reward_collapse)

        # batch para el calibrador
        X_list: List[np.ndarray] = []
        Y_list: List[float] = []

        for st in steps:
            p_base, _ = self._compute_penalty_base(st["s"], st["a"])
            feat_vec = np.array(
                [
                    p_base,
                    st["step"] / max(num_steps - 1, 1),
                    st["H_policy"],
                    st["temp"],
                    st["diversity"],
                    mean_r,
                ],
                dtype=np.float32,
            )
            X_list.append(feat_vec)
            Y_list.append(risk_label)

        X = torch.tensor(np.stack(X_list, axis=0), device=self.device)
        y = torch.tensor(np.array(Y_list, dtype=np.float32), device=self.device)

        # entrenamiento simple de regresión
        self.opt_calib.zero_grad()
        pred = self.calibrator(X)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        self.opt_calib.step()

        # refrescar distancias del grafo cada cierto número de episodios
        if self.episode_counter % self.cfg.fw_interval_episodes == 0:
            self._refresh_distances_if_needed()

        # limpiar buffer
        self.current_episode_steps = []
