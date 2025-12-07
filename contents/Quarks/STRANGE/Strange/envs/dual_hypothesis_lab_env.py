# ======================================================================
# DualHypothesisLabEnv
# ----------------------------------------------------------------------
# Entorno donde STRANGE y ANTI-STRANGE convergen sobre el MISMO estado.
#
# - phase:   0 = meseta, 1 = ambigua, 2 = claridad profunda
# - progress:  superficie (0..1)
# - stability: superficie (0..1)
# - latent_potential: nivel "subterráneo" (0..1) que anti-STRANGE excava
#
# STRANGE actúa sobre progress/stability (como HypothesisLabEnv).
# ANTI-STRANGE actúa sobre latent_potential y mete ruido estructurado
# que puede desviar el curso cuando hay mucha energía latente.
#
# Ambos reciben el MISMO reward global.
# ======================================================================

from dataclasses import dataclass
import numpy as np


@dataclass
class DualHypothesisState:
    phase: int
    progress: float
    stability: float
    latent_potential: float
    last_action_strange: int
    last_action_anti: int


class DualHypothesisLabEnv:
    """
    Entorno de convergencia STRANGE + ANTI-STRANGE.

    step(a_strange, a_anti) -> (next_state_id, reward, done, info)

    - Usa las tablas de HypothesisLabEnv para el efecto "superficie"
      (progress, stability) controlado por STRANGE.
    - Usa las tablas de AntiHypothesisLabEnv para el efecto "subsuelo"
      (latent_potential) controlado por ANTI-STRANGE.
    - La fase puede saltar a 2 cuando la latent_potential supera un
      umbral, simulando "descubrimiento profundo".
    """

    def __init__(
        self,
        max_steps: int = 50,
        phase_drift_prob: float = 0.04,
        potential_threshold: float = 0.6,
        seed: int = 0,
    ):
        self.max_steps = max_steps
        self.phase_drift_prob = phase_drift_prob
        self.potential_threshold = potential_threshold
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.state: DualHypothesisState | None = None

    # Ambos hadrones comparten el mismo espacio de acción (0..3)
    @property
    def n_actions(self) -> int:
        return 4

    # ----------------- Utils de estado -----------------

    def _encode_state(self, s: DualHypothesisState) -> int:
        """
        Codificación discreta:
        3 fases x 5 bins de progreso x 5 de estabilidad x 3 de potencial
        = 225 estados.
        """
        p_bin = int(np.clip(s.progress * 5, 0, 4))
        st_bin = int(np.clip(s.stability * 5, 0, 4))
        pot_bin = int(np.clip(s.latent_potential * 3, 0, 2))
        return s.phase * 75 + p_bin * 15 + st_bin * 3 + pot_bin

    def _sample_next_phase(self, phase: int, latent_potential: float) -> int:
        """
        Regla de fase híbrida:

        - Si la energía latente cruza el umbral, saltamos a fase 2.
        - Si no, hacemos drift local como en HypothesisLabEnv.
        """
        if latent_potential >= self.potential_threshold and phase < 2:
            return 2

        if self.rng.random() > self.phase_drift_prob:
            return phase

        delta = self.rng.integers(-1, 2)
        next_phase = int(np.clip(phase + delta, 0, 2))
        return next_phase

    # ----------------- Reset -----------------

    def reset(self) -> int:
        self.step_count = 0
        phase = int(self.rng.integers(0, 2))  # meseta / ambigua
        progress = 0.1
        stability = 0.2
        latent_potential = 0.0
        self.state = DualHypothesisState(
            phase=phase,
            progress=progress,
            stability=stability,
            latent_potential=latent_potential,
            last_action_strange=-1,
            last_action_anti=-1,
        )
        return self._encode_state(self.state)

    # ----------------- Dinámica núcleo -----------------

    def _surface_deltas(self, phase: int, a_strange: int):
        """
        Dinámica "superficie" copiada de HypothesisLabEnv,
        gobernada por la acción de STRANGE.
        """
        # d_progress, d_stability, base_reward
        if phase == 0:   # meseta
            if a_strange == 0:
                return 0.10, -0.02, 0.05
            elif a_strange == 1:
                return 0.03, 0.02, 0.02
            elif a_strange == 2:
                return 0.07, 0.00, 0.04
            else:
                return 0.01, 0.01, 0.00
        elif phase == 1:  # ambigua
            if a_strange == 0:
                return 0.04, -0.03, 0.01
            elif a_strange == 1:
                return 0.05, 0.06, 0.05
            elif a_strange == 2:
                return 0.03, 0.03, 0.04
            else:
                return 0.02, 0.01, 0.02
        else:  # claridad
            if a_strange == 0:
                return 0.03, -0.04, 0.00
            elif a_strange == 1:
                return 0.04, 0.04, 0.05
            elif a_strange == 2:
                return 0.03, 0.02, 0.03
            else:
                return 0.06, 0.03, 0.08

    def _latent_deltas(self, phase: int, a_anti: int):
        """
        Dinámica "subsuelo" basada en AntiHypothesisLabEnv.
        Devuelve d_progress_extra, d_stability_extra, d_potential, base_r_anti
        (los dos primeros van con peso bajo).
        """
        if phase == 0:
            if a_anti == 0:
                return 0.10, -0.03, 0.08, 0.02
            elif a_anti == 1:
                return 0.03, 0.02, 0.01, 0.03
            elif a_anti == 2:
                return 0.08, 0.00, 0.06, 0.03
            else:
                return 0.02, 0.02, 0.00, 0.02
        elif phase == 1:
            if a_anti == 0:
                return 0.03, -0.04, 0.06, 0.00
            elif a_anti == 1:
                return 0.05, 0.06, 0.01, 0.06
            elif a_anti == 2:
                return 0.04, 0.03, 0.05, 0.05
            else:
                return 0.06, 0.06, 0.01, 0.08
        else:
            if a_anti == 0:
                return 0.02, -0.05, -0.02, 0.00
            elif a_anti == 1:
                return 0.05, 0.05, -0.01, 0.07
            elif a_anti == 2:
                return 0.05, 0.03, -0.005, 0.06
            else:
                return 0.08, 0.05, -0.03, 0.12

    # ----------------- Step conjunto -----------------

    def step(self, a_strange: int, a_anti: int):
        """
        Paso conjunto:

        - STRANGE elige a_strange.
        - ANTI-STRANGE elige a_anti.
        - El entorno combina ambas influencias y devuelve:
            next_state_id, reward_global, done, info
        """
        assert self.state is not None, "Llama a reset() antes de step()"
        self.step_count += 1
        s = self.state
        phase = s.phase

        # Dinámica de superficie (STRANGE)
        dP_s, dS_s, base_r_s = self._surface_deltas(phase, a_strange)

        # Dinámica de subsuelo (ANTI-STRANGE)
        dP_a, dS_a, dPot, base_r_a = self._latent_deltas(phase, a_anti)

        # Mezcla: STRANGE domina la superficie, ANTI añade un 30% de ruido estructurado
        d_progress = dP_s + 0.3 * dP_a
        d_stability = dS_s + 0.3 * dS_a

        # Ruido leve
        d_progress += self.rng.normal(0.0, 0.01)
        d_stability += self.rng.normal(0.0, 0.01)

        new_progress = float(np.clip(s.progress + d_progress, 0.0, 1.0))
        new_stability = float(np.clip(s.stability + d_stability, 0.0, 1.0))
        new_pot = float(np.clip(s.latent_potential + dPot, 0.0, 1.0))

        # Reward global:
        #   - Superficie (como HypothesisLabEnv)
        #   - Potencial latente
        #   - Sinergia cuando las acciones son distintas y hay potencial alto
        reward_surface = base_r_s + 0.4 * new_progress + 0.3 * new_stability
        reward_latent = 0.2 * new_pot + 0.3 * base_r_a

        synergy = 0.0
        if a_strange != a_anti and new_pot > 0.5:
            synergy += 0.03
        if a_strange == a_anti and new_pot < 0.2:
            # si ambos se quedan pegados a lo mismo con potencial bajo: castigo
            synergy -= 0.02

        reward = reward_surface + reward_latent + synergy

        # Penalización suave por repetir exactamente la misma acción
        if a_strange == s.last_action_strange:
            reward -= 0.01
        if a_anti == s.last_action_anti:
            reward -= 0.01

        # Penalización si todo está hundido
        if new_progress < 0.15 and new_stability < 0.15 and new_pot < 0.15:
            reward -= 0.08

        # Actualizar fase
        next_phase = self._sample_next_phase(phase, new_pot)

        # Actualizar estado
        self.state = DualHypothesisState(
            phase=next_phase,
            progress=new_progress,
            stability=new_stability,
            latent_potential=new_pot,
            last_action_strange=a_strange,
            last_action_anti=a_anti,
        )

        done = self.step_count >= self.max_steps

        info = {
            "phase": next_phase,
            "progress": new_progress,
            "stability": new_stability,
            "latent_potential": new_pot,
        }

        return self._encode_state(self.state), float(reward), bool(done), info
