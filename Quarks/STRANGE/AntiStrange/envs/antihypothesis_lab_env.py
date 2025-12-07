# ======================================================================
# ANTI-HYPOTHESIS LAB ENV
# Entorno diseñado para forzar exploración disruptiva (ANTI-STRANGE)
# ======================================================================

from dataclasses import dataclass
import numpy as np

@dataclass
class AntiHypothesisState:
    phase: int
    progress: float
    stability: float
    latent_potential: float
    last_action: int


class AntiHypothesisLabEnv:
    """
    Variante del laboratorio diseñada para exponer el comportamiento del
    agente ANTI-STRANGE: romper attractores y desbloquear recompensas
    profundas mediante exploración insistente.
    """

    def __init__(
        self,
        max_steps: int = 50,
        phase_drift_prob: float = 0.03,
        seed: int = 0,
        potential_threshold: float = 0.6,
    ):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.phase_drift_prob = phase_drift_prob
        self.potential_threshold = potential_threshold
        self.step_count = 0
        self.state = None

    @property
    def n_actions(self):
        return 4   # respeta el mismo espacio de acción

    def reset(self):
        self.step_count = 0
        self.state = AntiHypothesisState(
            phase=1,
            progress=0.1,
            stability=0.2,
            latent_potential=0.0,
            last_action=-1,
        )
        return self._encode_state(self.state)

    def _encode_state(self, s):
        p_bin  = int(np.clip(s.progress * 5, 0, 4))
        s_bin  = int(np.clip(s.stability * 5, 0, 4))
        pot_bin = int(np.clip(s.latent_potential * 3, 0, 2))
        return s.phase * 75 + p_bin * 15 + s_bin * 3 + pot_bin

    def _sample_next_phase(self, phase, potential):
        if potential >= self.potential_threshold and phase < 2:
            return 2
        if self.rng.random() > self.phase_drift_prob:
            return phase
        delta = self.rng.integers(-1, 2)
        return int(np.clip(phase + delta, 0, 2))

    def step(self, action: int):
        self.step_count += 1
        s = self.state

        phase = s.phase

        dP = dS = dPot = 0.0

        if phase == 0:
            if action == 0:
                dP, dS, r = 0.10, -0.03, 0.02
                dPot = 0.08
            elif action == 1:
                dP, dS, r = 0.03, 0.02, 0.03
                dPot = 0.01
            elif action == 2:
                dP, dS, r = 0.08, 0.00, 0.03
                dPot = 0.06
            else:
                dP, dS, r = 0.02, 0.02, 0.02

        elif phase == 1:
            if action == 0:
                dP, dS, r = 0.03, -0.04, 0.00
                dPot = 0.06
            elif action == 1:
                dP, dS, r = 0.05, 0.06, 0.06
                dPot = 0.01
            elif action == 2:
                dP, dS, r = 0.04, 0.03, 0.05
                dPot = 0.05
            else:
                dP, dS, r = 0.06, 0.06, 0.08

        else:
            if action == 0:
                dP, dS, r = 0.02, -0.05, 0.0
                dPot = -0.02
            elif action == 1:
                dP, dS, r = 0.05, 0.05, 0.07
                dPot = -0.01
            elif action == 2:
                dP, dS, r = 0.05, 0.03, 0.06
                dPot = -0.005
            else:
                dP, dS, r = 0.08, 0.05, 0.12
                dPot = -0.03

        dP += self.rng.normal(0, 0.01)
        dS += self.rng.normal(0, 0.01)

        new_p = np.clip(s.progress + dP, 0, 1)
        new_s = np.clip(s.stability + dS, 0, 1)
        new_pot = np.clip(s.latent_potential + dPot, 0, 1)

        reward = r + 0.35 * new_p + 0.30 * new_s

        if action == s.last_action:
            reward -= 0.02

        if new_p < 0.15 and new_s < 0.15:
            reward -= 0.08

        next_phase = self._sample_next_phase(phase, new_pot)

        self.state = AntiHypothesisState(
            phase=next_phase,
            progress=new_p,
            stability=new_s,
            latent_potential=new_pot,
            last_action=action,
        )

        done = self.step_count >= self.max_steps

        info = {
            "phase": next_phase,
            "progress": float(new_p),
            "stability": float(new_s),
            "latent_potential": float(new_pot),
        }

        return self._encode_state(self.state), float(reward), bool(done), info
