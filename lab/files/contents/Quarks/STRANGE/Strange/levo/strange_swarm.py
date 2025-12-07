"""
STRANGE Swarm
-------------
Implementa:

- HF-Levo tabular para cada agente del swarm.
- Control termodinámico T con dT/dt = E - R.
- Proto-memoria por estado.
- Gate w_t que mezcla p_swarm y proto, dando la política final.

Este archivo no depende de un entorno específico; sólo asume:
- estados discretos hashables (por ejemplo enteros),
- acciones indexadas 0..n_actions-1.
"""

from dataclasses import dataclass
from typing import Dict, List, Hashable
import numpy as np
import math


# ===================== HF-Levo agent =====================

@dataclass
class HFLevoAgent:
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.99
    tau_base: float = 1.0

    def __post_init__(self):
        # Q: dict[state][action] -> value
        self.Q: Dict[Hashable, np.ndarray] = {}
        # Fase fija por acción para inducir diversidad real entre acciones
        self.action_phases = np.linspace(
            0.0, 2 * math.pi, self.n_actions, endpoint=False
        )

    def _ensure_state(self, s: Hashable):
        if s not in self.Q:
            self.Q[s] = np.zeros(self.n_actions, dtype=float)

    def policy(
        self,
        s: Hashable,
        t: int,
        T: float,
        hf_amp: float,
        hf_omega: float,
        phase_offset: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Devuelve distribución de prob de acciones usando HF-Levo:
            scores = Q + A(T) * cos(w(T)*t + phi_h,a)
            p ~ softmax(scores / tau(T))
        """
        self._ensure_state(s)
        q = self.Q[s]

        # temperatura de softmax: tau sube con T (más exploración)
        tau = self.tau_base * (1.0 + 0.5 * T)

        # oscilación por acción: desfase del agente + desfase propio por acción
        osc_vec = hf_amp * np.cos(
            hf_omega * t + phase_offset + self.action_phases
        )
        scores = q + osc_vec

        # softmax numéricamente estable
        z = scores - scores.max()
        exp_z = np.exp(z / max(1e-8, tau))
        p = exp_z / exp_z.sum()
        return p

    def update(
        self,
        s: Hashable,
        a: int,
        r: float,
        s_next: Hashable,
        done: bool,
    ):
        self._ensure_state(s)
        self._ensure_state(s_next)
        q = self.Q[s]
        q_next = self.Q[s_next]
        target = r if done else (r + self.gamma * float(q_next.max()))
        td = target - q[a]
        q[a] += self.alpha * td


# ===================== Config STRANGE =====================

@dataclass
class StrangeConfig:
    n_agents: int = 5
    n_actions: int = 4
    hf_amp_base: float = 0.15
    hf_omega_base: float = 0.35

    T_init: float = 0.7
    T_min: float = 0.0
    T_max: float = 2.0
    eta_T: float = 0.05  # pasos de T más suaves

    # umbrales E (excitación)
    eps_R: float = 0.02      # estancamiento si |ΔR| pequeño
    H_min: float = 0.7       # baja entropía = poca exploración
    D_min: float = 0.10      # poca diversidad de políticas

    # umbrales R (regulación)
    S_max: float = 0.15      # strangeness muy alta
    H_max: float = 1.25      # entropía casi máxima (~log(4)=1.386)
    eps_crash: float = 0.02  # caída fuerte de reward

    # gates (controlan cuánto pesa la memoria)
    T_mid: float = 1.1
    S_mid: float = 0.08
    H_mid: float = 1.0

    kT: float = 1.5
    kS: float = 2.0
    kH: float = 1.0

    proto_alpha: float = 0.1

    seed: int = 0


# ===================== STRANGE Swarm =====================

class StrangeSwarm:
    """
    Implementación de alto nivel del hadrón STRANGE.

    Uso:
        swarm = StrangeSwarm(config)
        s = env.reset()
        for t in ...:
            action, info = swarm.act(s, t)
            s_next, r, done, env_info = env.step(action)
            swarm.update(s, action, r, s_next, done)
    """
    def __init__(self, config: StrangeConfig):
        self.cfg = config
        self.n_actions = config.n_actions
        self.agents: List[HFLevoAgent] = [
            HFLevoAgent(n_actions=config.n_actions)
            for _ in range(config.n_agents)
        ]
        self.rng = np.random.default_rng(config.seed)

        # fases internas
        self.T = config.T_init
        self.proto: Dict[Hashable, np.ndarray] = {}

        # para métricas
        self.last_reward = 0.0
        self.last_delta_R = 0.0  # ΔR del paso previo

        # fases diferentes para cada agente
        self.agent_phases = np.linspace(
            0.0, 2 * math.pi, config.n_agents, endpoint=False
        )

    # --------- Métricas internas ---------
    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        p_safe = np.clip(p, 1e-8, 1.0)
        return float(-(p_safe * np.log(p_safe)).sum())

    @staticmethod
    def _diversity(ps: np.ndarray) -> float:
        # media de distancias L1 entre políticas
        H, A = ps.shape
        if H <= 1:
            return 0.0
        d = 0.0
        cnt = 0
        for i in range(H):
            for j in range(i + 1, H):
                d += float(np.abs(ps[i] - ps[j]).mean())
                cnt += 1
        return d / max(1, cnt)

    # --------- Core ---------
    def _compute_swarm_policy(self, s: Hashable, t: int):
        cfg = self.cfg
        ps = []
        for h, agent in enumerate(self.agents):
            phase = self.agent_phases[h]
            p_h = agent.policy(
                s,
                t,
                self.T,
                hf_amp=cfg.hf_amp_base * (1.0 + 0.5 * self.T),
                hf_omega=cfg.hf_omega_base * (1.0 + 0.25 * self.T),
                phase_offset=phase,
                rng=self.rng,
            )
            ps.append(p_h)
        ps = np.stack(ps)  # [H, A]
        p_swarm = ps.mean(axis=0)
        var = ps.var(axis=0)
        S_t = float(var.mean())
        H_swarm = self._entropy(p_swarm)
        diversity = self._diversity(ps)
        return p_swarm, S_t, H_swarm, diversity

    def _update_proto(self, s: Hashable, p_swarm: np.ndarray):
        cfg = self.cfg
        if s not in self.proto:
            self.proto[s] = p_swarm.copy()
        else:
            self.proto[s] = (
                (1.0 - cfg.proto_alpha) * self.proto[s]
                + cfg.proto_alpha * p_swarm
            )

    def _thermo_step(
        self,
        delta_R: float,
        H_swarm: float,
        S_t: float,
        diversity: float,
    ):
        cfg = self.cfg
        # Excitación
        E_t = (
            (abs(delta_R) < cfg.eps_R) * 1.0 +
            (H_swarm < cfg.H_min) * 1.0 +
            (diversity < cfg.D_min) * 1.0
        )
        # Regulación
        R_t = (
            (S_t > cfg.S_max) * 1.0 +
            (H_swarm > cfg.H_max) * 1.0 +
            (delta_R < -cfg.eps_crash) * 1.0
        )
        self.T += cfg.eta_T * (E_t - R_t)
        self.T = float(np.clip(self.T, cfg.T_min, cfg.T_max))
        return float(E_t), float(R_t)

    def _gate(self, H_swarm: float, S_t: float) -> float:
        cfg = self.cfg
        x = (
            cfg.kT * (self.T - cfg.T_mid) +
            cfg.kS * (S_t - cfg.S_mid) +
            cfg.kH * (H_swarm - cfg.H_mid)
        )
        # sigmoide
        w = 1.0 / (1.0 + math.exp(-x))
        return float(w)

    def act(self, s: Hashable, t: int):
        """
        Calcula la política STRANGE completa y samplea una acción.
        Devuelve:
            action, info_dict
        """
        # política del swarm y métricas internas
        p_swarm, S_t, H_swarm, diversity = self._compute_swarm_policy(s, t)
        self._update_proto(s, p_swarm)

        # usamos ΔR del paso anterior para la termodinámica
        delta_R = self.last_delta_R
        E_t, R_t = self._thermo_step(delta_R, H_swarm, S_t, diversity)

        # gate memoria vs swarm
        w_t = self._gate(H_swarm, S_t)
        proto_s = self.proto[s]
        p_final = (1.0 - w_t) * p_swarm + w_t * proto_s
        p_final = p_final / p_final.sum()

        # sample acción
        a = int(self.rng.choice(len(p_final), p=p_final))

        info = {
            "T": self.T,
            "S_t": S_t,
            "H_swarm": H_swarm,
            "diversity": diversity,
            "E_t": E_t,
            "R_t": R_t,
            "w_t": w_t,
            "p_swarm": p_swarm,
            "p_final": p_final,
        }
        return a, info

    def update(
        self,
        s: Hashable,
        a: int,
        r: float,
        s_next: Hashable,
        done: bool,
    ):
        # Actualizamos Q de todos los agentes (paralelo tabular compartiendo experiencia)
        for agent in self.agents:
            agent.update(s, a, r, s_next, done)

        # ΔR real respecto al paso anterior
        delta_R = r - self.last_reward
        self.last_reward = r
        self.last_delta_R = delta_R

        return delta_R
