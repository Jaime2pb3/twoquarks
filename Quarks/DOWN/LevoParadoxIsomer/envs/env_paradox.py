
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


@dataclass(frozen=True)
class ParadoxState:
    """Structured view of the epistemic state."""
    ambiguity_type: int  # 0..2
    evidence_bin: int    # 0..2
    risk_level: int      # 0..1
    phase: int           # 1..3


class EpistemicValleyEnv:
    """
    Epistemic Valley Inframe v2

    Discrete one-step environment modelling epistemic risk under ambiguity.

    State s = (ambiguity_type, evidence_bin, risk_level, phase)
      - ambiguity_type: 0=lexical, 1=scope, 2=missing-data
      - evidence_bin:   0=low, 1=medium, 2=high
      - risk_level:     0=low, 1=high
      - phase:          1, 2, or 3 (reward regime)

    Actions:
      0 = HOLD      (stay with current uncertainty)
      1 = EXPAND    (seek more evidence / broader context)
      2 = CLARIFY   (ask a targeted clarification)
      3 = DEFER     (explicitly postpone the decision)
      4 = ANSWER    (commit to an answer)

    Episodes are single-step: reset() -> step(action) -> done=True.
    """
    # actions (kept as attributes so agents can introspect)
    HOLD = 0
    EXPAND = 1
    CLARIFY = 2
    DEFER = 3
    ANSWER = 4

    def __init__(self, phase: int = 1, seed: Optional[int] = None) -> None:
        assert phase in (1, 2, 3), "phase must be 1, 2 or 3"
        self.ambiguity_types = ["lexical", "scope", "missing-data"]
        self.evidence_bins = ["low", "medium", "high"]
        self.risk_levels = ["low", "high"]
        self.phases = [1, 2, 3]

        self.n_ambiguity = len(self.ambiguity_types)
        self.n_evidence = len(self.evidence_bins)
        self.n_risk = len(self.risk_levels)
        self.n_phase = len(self.phases)

        self.n_states = self.n_ambiguity * self.n_evidence * self.n_risk * self.n_phase
        self.n_actions = 5

        self.phase = phase
        self.rng = np.random.default_rng(seed)
        self.state: Optional[ParadoxState] = None
        self.done: bool = False

    # ------------------------------------------------------------------
    # state indexing helpers
    # ------------------------------------------------------------------
    def encode_state(self, s: ParadoxState) -> int:
        """Flatten ParadoxState into [0, n_states)."""
        idx = s.ambiguity_type
        idx = idx * self.n_evidence + s.evidence_bin
        idx = idx * self.n_risk + s.risk_level
        # phase is 1..3, internally we use 0..2
        phase_index = s.phase - 1
        idx = idx * self.n_phase + phase_index
        return idx

    def decode_state(self, idx: int) -> ParadoxState:
        """Inverse of encode_state."""
        phase_index = idx % self.n_phase
        idx //= self.n_phase
        risk_level = idx % self.n_risk
        idx //= self.n_risk
        evidence_bin = idx % self.n_evidence
        idx //= self.n_evidence
        ambiguity_type = idx
        return ParadoxState(
            ambiguity_type=int(ambiguity_type),
            evidence_bin=int(evidence_bin),
            risk_level=int(risk_level),
            phase=int(phase_index + 1),
        )

    # ------------------------------------------------------------------
    # core env API
    # ------------------------------------------------------------------
    def _sample_state(self) -> ParadoxState:
        a = int(self.rng.integers(0, self.n_ambiguity))
        e = int(self.rng.integers(0, self.n_evidence))
        r = int(self.rng.integers(0, self.n_risk))
        return ParadoxState(
            ambiguity_type=a,
            evidence_bin=e,
            risk_level=r,
            phase=self.phase,
        )

    def reset(self) -> Tuple[int, Dict]:
        """Sample a new epistemic state and return its index + info."""
        self.state = self._sample_state()
        self.done = False
        s_idx = self.encode_state(self.state)
        return s_idx, {"state": self.state}

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """Apply one action, return (next_state_index, reward, done, info)."""
        assert self.state is not None, "Call reset() before step()."
        s = self.state
        reward, failure_mode = self._reward(s, action)
        self.done = True
        info = {
            "state": s,
            "failure_mode": failure_mode,
        }
        # one-step episode; next state is irrelevant for this env
        return self.encode_state(s), float(reward), self.done, info

    # ------------------------------------------------------------------
    # reward function
    # ------------------------------------------------------------------
    def _reward(self, s: ParadoxState, action: int) -> Tuple[float, str]:
        """
        Reward shaped by:
          - evidence / risk alignment
          - failure mode (overconfident vs overcautious)
          - phase (reward regime and noise)

        We implement a clean, reproducible version matching the narrative in
        the original MODELS document: aggressive actions are good when
        evidence is strong and risk is low, dangerous when evidence is weak
        and risk is high, and conservative actions behave inversely.
        """
        e_str = self.evidence_bins[s.evidence_bin]
        r_str = self.risk_levels[s.risk_level]
        phase = s.phase

        high_risk = (r_str == "high")
        low_risk = (r_str == "low")
        low_evidence = (e_str == "low")
        high_evidence = (e_str == "high")

        failure_mode = "neutral"
        base = 0.0

        # --- classify aggressive vs conservative actions ---
        aggressive = action in (self.EXPAND, self.ANSWER)
        conservative = action in (self.HOLD, self.DEFER)

        # --- base reward logic before phase-specific shaping ---
        if aggressive and high_risk and low_evidence:
            # archetypal overconfidence: answer aggressively with little support
            base = -2.0
            failure_mode = "overconfident"
        elif conservative and low_risk and high_evidence:
            # archetypal overcautious: you could answer safely but you freeze
            base = -0.8
            failure_mode = "overcautious"
        else:
            # reasonably aligned choices
            if aggressive and high_evidence and low_risk:
                base = +2.0
            elif aggressive and not (high_risk and low_evidence):
                base = +0.8
            elif conservative and high_risk and low_evidence:
                base = +1.2
            else:
                base = 0.0

        # --- phase-specific shaping ---
        if phase == 1:
            # Slight bonus for successful aggressive behaviour
            if aggressive and base > 0:
                base += 0.5
        elif phase == 2:
            # Harsher overconfidence penalty
            if failure_mode == "overconfident":
                base -= 1.0
        elif phase == 3:
            # Noisy feedback regime
            noise = float(self.rng.normal(0.0, 0.5))
            base += noise
            base = float(np.clip(base, -3.0, 3.0))

        return base, failure_mode
