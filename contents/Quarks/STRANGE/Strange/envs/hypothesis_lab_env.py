
"""
HypothesisLabEnv
-----------------
Entorno abstracto tipo "laboratorio de hipótesis".

- El agente decide entre acciones de exploración, refinamiento y consolidación.
- El entorno tiene 3 fases (regímenes) que cambian con el tiempo:
    0: Meseta (stuck)        -> explorar ayuda más.
    1: Zona ambigua (ruido)  -> combinar / refinar ayuda.
    2: Claridad (explotar)   -> consolidar la hipótesis principal.

Estado (discreto):
    (phase, progress_bin, stability_bin)
que se empaqueta en un entero para que sea fácil usar tabular Q.
"""

import numpy as np
from dataclasses import dataclass

ACTIONS = {
    0: "explore_new",      # probar experimento radical nuevo
    1: "refine_current",   # mejorar hipótesis actual
    2: "combine_paths",    # combinar líneas previas
    3: "exploit_main"      # consolidar y explotar la mejor
}
N_ACTIONS = len(ACTIONS)


@dataclass
class HypothesisState:
    phase: int          # 0,1,2
    progress: float     # [0,1]
    stability: float    # [0,1] (qué tan claros son los resultados)


class HypothesisLabEnv:
    """
    Entorno minimalista pero no trivial.

    Reward intuitivo:
    - Fase 0 (meseta): explorar/combine mejoran más el progreso.
    - Fase 1 (ambigua): refine/combine ayudan a subir estabilidad.
    - Fase 2 (claridad): exploit/refine consolidan y dan mayor reward.

    Además, el entorno cambia de fase con una dinámica lenta, para
    forzar al swarm STRANGE a detectar el régimen mediante métricas
    internas, no observables explícitos.
    """
    def __init__(self,
                 max_steps: int = 50,
                 phase_drift_prob: float = 0.05,
                 seed: int = 0):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.phase_drift_prob = phase_drift_prob
        self.step_count = 0
        self.state = None

    @property
    def n_actions(self):
        return N_ACTIONS

    def _sample_next_phase(self, phase: int) -> int:
        """Cambia de fase con prob pequeña y movimiento local."""
        if self.rng.random() > self.phase_drift_prob:
            return phase
        # moverse -1,0,+1 pero quedando en 0..2
        delta = self.rng.integers(-1, 2)
        next_phase = int(np.clip(phase + delta, 0, 2))
        return next_phase

    def reset(self):
        self.step_count = 0
        # empezamos casi siempre en meseta-ambigua
        phase = int(self.rng.integers(0, 2))
        progress = 0.1
        stability = 0.2
        self.state = HypothesisState(phase, progress, stability)
        return self._encode_state(self.state)

    def _encode_state(self, s: HypothesisState) -> int:
        # 3 fases, 5 bins de progreso, 5 de estabilidad -> 75 estados
        p_bin = int(np.clip(s.progress * 5, 0, 4))
        st_bin = int(np.clip(s.stability * 5, 0, 4))
        return s.phase * 25 + p_bin * 5 + st_bin

    def step(self, action: int):
        """
        Devuelve: next_state_id, reward, done, info.
        """
        assert 0 <= action < N_ACTIONS
        self.step_count += 1
        s = self.state

        # Efecto base de la acción sobre progreso y estabilidad
        d_progress = 0.0
        d_stability = 0.0

        # Matriz efecto (phase, action) -> (d_prog, d_stab, base_reward)
        # Valores hechos a mano para inducir multi-modalidad.
        phase = s.phase
        if phase == 0:   # meseta
            if action == 0:  # explore_new
                d_progress, d_stability, base_r = 0.10, -0.02, 0.05
            elif action == 1:  # refine
                d_progress, d_stability, base_r = 0.03, 0.02, 0.02
            elif action == 2:  # combine
                d_progress, d_stability, base_r = 0.07, 0.00, 0.04
            else:  # exploit
                d_progress, d_stability, base_r = 0.01, 0.01, 0.0
        elif phase == 1:  # ambigua
            if action == 0:
                d_progress, d_stability, base_r = 0.04, -0.03, 0.01
            elif action == 1:
                d_progress, d_stability, base_r = 0.05, 0.06, 0.05
            elif action == 2:
                d_progress, d_stability, base_r = 0.03, 0.03, 0.04
            else:  # exploit
                d_progress, d_stability, base_r = 0.02, 0.01, 0.02
        else:  # phase == 2, claridad
            if action == 0:
                d_progress, d_stability, base_r = 0.03, -0.04, 0.0
            elif action == 1:
                d_progress, d_stability, base_r = 0.04, 0.04, 0.05
            elif action == 2:
                d_progress, d_stability, base_r = 0.03, 0.02, 0.03
            else:  # exploit
                d_progress, d_stability, base_r = 0.06, 0.03, 0.08

        # Ruido leve
        d_progress += self.rng.normal(0.0, 0.01)
        d_stability += self.rng.normal(0.0, 0.01)

        new_progress = float(np.clip(s.progress + d_progress, 0.0, 1.0))
        new_stability = float(np.clip(s.stability + d_stability, 0.0, 1.0))

        # Reward combina:
        # - progreso
        # - estabilidad alta
        # - penalización si todo muy bajo
        reward = base_r
        reward += 0.4 * new_progress + 0.3 * new_stability
        if new_progress < 0.2 and new_stability < 0.2:
            reward -= 0.1

        next_phase = self._sample_next_phase(phase)
        self.state = HypothesisState(next_phase, new_progress, new_stability)
        done = self.step_count >= self.max_steps

        info = {
            "phase": next_phase,
            "progress": new_progress,
            "stability": new_stability
        }
        return self._encode_state(self.state), float(reward), bool(done), info
