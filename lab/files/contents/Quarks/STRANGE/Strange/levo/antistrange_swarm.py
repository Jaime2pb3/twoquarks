"""
ANTI-STRANGE Swarm
------------------
Contraparte "antiquark" del hadrón STRANGE.

Diferencias conceptuales con StrangeSwarm:

- Misma base HF-Levo tabular, pero:
  * Termodinámica responde distinto a ΔR (premia mejoras fuertes).
  * Gate w_t se activa cuando el sistema está demasiado ordenado.
  * Usa una memoria *repulsora*: anti-proto, que empuja a acciones
    históricamente poco visitadas (huecos del mapa de políticas).

- Interfaz compatible con StrangeSwarm:
    AntiStrangeConfig, AntiStrangeSwarm.act(), AntiStrangeSwarm.update().
"""

from dataclasses import dataclass
from typing import Dict, List, Hashable
import numpy as np
import math


# ===================== HF-Levo agent (copiado para independencia) =====================

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
        HF-Levo:
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


# ===================== Config ANTI-STRANGE =====================

@dataclass
class AntiStrangeConfig:
    n_agents: int = 5
    n_actions: int = 4
    hf_amp_base: float = 0.15
    hf_omega_base: float = 0.35

    T_init: float = 0.7
    T_min: float = 0.0
    T_max: float = 2.0
    eta_T: float = 0.05  # pasos de T

    # --- Excitación (sube T) ---
    # Queremos subir T cuando:
    # - Hay mejoras fuertes (|ΔR| grande positiva).
    # - El sistema está demasiado ordenado (baja entropía y baja diversidad).
    eps_R_pos: float = 0.03    # mejora significativa de recompensa
    H_low: float = 0.7         # entropía baja
    D_low: float = 0.08        # diversidad baja

    # --- Regulación (baja T) ---
    # Queremos bajar T cuando:
    # - El sistema se vuelve caótico (H o diversidad altas).
    # - Hay caídas fuertes de recompensa.
    H_high: float = 1.25       # entropía muy alta (~max)
    D_high: float = 0.20       # diversidad muy alta
    eps_crash: float = 0.03    # caída fuerte de reward

    # --- Gate memoria repulsora ---
    # Gate alto cuando el sistema está demasiado "congelado":
    # T baja, S_t baja, H_swarm baja.
    T_mid: float = 0.9
    S_mid: float = 0.06
    H_mid: float = 0.8

    kT: float = 1.5
    kS: float = 2.0
    kH: float = 1.0

    proto_alpha: float = 0.1

    seed: int = 1  # distinto de Strange por defecto


# ===================== ANTI-STRANGE Swarm =====================

class AntiStrangeSwarm:
    """
    Hadron ANTI-STRANGE.

    Diferencias clave vs StrangeSwarm:
      - T sube cuando hay mejoras fuertes y orden excesivo.
      - Gate w_t empuja a memoria REPULSORA (anti-proto).
      - Diseñado para explorar huecos de la política del enjambre.

    Interfaz:
        anti = AntiStrangeSwarm(cfg)
        s = env.reset()
        for t in ...:
            a, info = anti.act(s, t)
            s_next, r, done, env_info = env.step(a)
            anti.update(s, a, r, s_next, done)
    """

    def __init__(self, config: AntiStrangeConfig):
        self.cfg = config
        self.n_actions = config.n_actions
        self.agents: List[HFLevoAgent] = [
            HFLevoAgent(n_actions=config.n_actions)
            for _ in range(config.n_agents)
        ]
        self.rng = np.random.default_rng(config.seed)

        # estado interno
        self.T = config.T_init
        self.proto: Dict[Hashable, np.ndarray] = {}

        self.last_reward = 0.0
        self.last_delta_R = 0.0

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

    # --------- Core swarm policy ---------
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

    def _anti_proto(self, s: Hashable) -> np.ndarray:
        """
        Construye memoria repulsora:
            anti_proto ∝ 1 - proto
        Es decir, favorece acciones históricamente poco usadas.
        """
        proto_s = self.proto[s]
        raw = 1.0 - proto_s
        raw = np.clip(raw, 1e-8, None)
        return raw / raw.sum()

    def _thermo_step(
        self,
        delta_R: float,
        H_swarm: float,
        S_t: float,
        diversity: float,
    ):
        """
        Control térmico "anti":

        - E_t (excitación, sube T):
            * |ΔR| grande positiva (mejora fuerte de recompensa).
            * Orden excesivo: H_swarm baja y diversidad baja.

        - R_t (regulación, baja T):
            * Caos fuerte: H_swarm muy alta o diversidad muy alta.
            * Crash de recompensa: ΔR muy negativa.
        """
        cfg = self.cfg

        improving = delta_R > cfg.eps_R_pos
        too_ordered = (H_swarm < cfg.H_low) and (diversity < cfg.D_low)

        E_t = (
            (1.0 if improving else 0.0) +
            (1.0 if too_ordered else 0.0)
        )

        too_chaotic = (H_swarm > cfg.H_high) or (diversity > cfg.D_high)
        crashing = delta_R < -cfg.eps_crash

        R_t = (
            (1.0 if too_chaotic else 0.0) +
            (1.0 if crashing else 0.0)
        )

        self.T += cfg.eta_T * (E_t - R_t)
        self.T = float(np.clip(self.T, cfg.T_min, cfg.T_max))
        return float(E_t), float(R_t)

    def _gate(self, H_swarm: float, S_t: float) -> float:
        """
        Gate invertido:

        - Queremos w_t alto cuando el sistema está demasiado
          ordenado y frío (baja entropía / baja varianza / T baja),
          para empujar hacia acciones en los "huecos" (anti-proto).

        - Cuando T, S_t y H_swarm son altos, w_t tiende a 0
          y se deja actuar más a p_swarm.
        """
        cfg = self.cfg
        x = (
            cfg.kT * (cfg.T_mid - self.T) +
            cfg.kS * (cfg.S_mid - S_t) +
            cfg.kH * (cfg.H_mid - H_swarm)
        )
        w = 1.0 / (1.0 + math.exp(-x))
        return float(w)

    # --------- Interfaz pública ---------
    def act(self, s: Hashable, t: int):
        """
        Calcula la política ANTI-STRANGE y samplea una acción.
        Devuelve:
            action, info_dict
        """
        p_swarm, S_t, H_swarm, diversity = self._compute_swarm_policy(s, t)
        self._update_proto(s, p_swarm)

        # feedback térmico con ΔR real del paso previo
        delta_R = self.last_delta_R
        E_t, R_t = self._thermo_step(delta_R, H_swarm, S_t, diversity)

        # memoria repulsora
        anti_proto_s = self._anti_proto(s)
        w_t = self._gate(H_swarm, S_t)

        p_final = (1.0 - w_t) * p_swarm + w_t * anti_proto_s
        p_final = p_final / p_final.sum()

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
        # Actualizamos Q de todos los agentes
        for agent in self.agents:
            agent.update(s, a, r, s_next, done)

        # ΔR real
        delta_R = r - self.last_reward
        self.last_reward = r
        self.last_delta_R = delta_R

        return delta_R
