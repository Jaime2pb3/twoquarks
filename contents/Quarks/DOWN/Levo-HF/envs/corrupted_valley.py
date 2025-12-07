import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np


@dataclass
class StepResult:
    obs: int
    reward: float
    done: bool
    info: Dict[str, Any]


class CorruptedValleyEnv:
    """
    7x7 gridworld with:
      - Start S at bottom-left (row=6, col=0)
      - Goal G at top-right (row=0, col=6)
      - Corrupted valley V in the center of the grid.

    Phases:
        1: Valley is falsely attractive (positive bonus).
        2: Valley is corrected (neutral or mildly negative).
        3: Same rewards as phase 2. Intended for noisy-training experiments.

    Observation is a single integer state index in [0, n_states).
    Actions: 0=up, 1=right, 2=down, 3=left.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        grid_size: int = 7,
        valley_reward_phase1: float = 2.0,
        valley_reward_phase2: float = -0.5,
        step_penalty: float = -0.01,
        max_steps: int = 50,
        phase: int = 1,
    ) -> None:
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4

        self.valley_reward_phase1 = valley_reward_phase1
        self.valley_reward_phase2 = valley_reward_phase2
        self.step_penalty = step_penalty
        self.max_steps = max_steps

        self.start_pos = (grid_size - 1, 0)
        self.goal_pos = (0, grid_size - 1)

        self.rng = np.random.default_rng(seed)
        self.phase = phase

        # center 3x3 block as valley
        c = grid_size // 2
        self.valley_cells = {
            (r, c2)
            for r in range(c - 1, c + 2)
            for c2 in range(c - 1, c + 2)
        }

        self.pos: Tuple[int, int] = self.start_pos
        self.steps = 0

    def _state_from_pos(self, pos: Tuple[int, int]) -> int:
        r, c = pos
        return r * self.grid_size + c

    def _obs(self) -> int:
        return self._state_from_pos(self.pos)

    def reset(self) -> int:
        self.pos = self.start_pos
        self.steps = 0
        return self._obs()

    def set_phase(self, phase: int) -> None:
        if phase not in (1, 2, 3):
            raise ValueError(f"Invalid phase: {phase}")
        self.phase = phase

    def step(self, action: int) -> StepResult:
        self.steps += 1

        r, c = self.pos
        if action == 0:      # up
            r -= 1
        elif action == 1:    # right
            c += 1
        elif action == 2:    # down
            r += 1
        elif action == 3:    # left
            c -= 1

        # clamp to grid
        r = max(0, min(self.grid_size - 1, r))
        c = max(0, min(self.grid_size - 1, c))
        self.pos = (r, c)

        reward = self.step_penalty
        done = False
        info: Dict[str, Any] = {}

        in_valley = self.pos in self.valley_cells
        info["valley"] = in_valley

        if self.pos == self.goal_pos:
            reward += 1.0
            done = True
            info["terminal"] = "goal"
        elif in_valley:
            if self.phase == 1:
                reward += self.valley_reward_phase1
            else:
                reward += self.valley_reward_phase2

        if self.steps >= self.max_steps and not done:
            done = True
            info["terminal"] = "timeout"

        return StepResult(
            obs=self._obs(),
            reward=reward,
            done=done,
            info=info,
        )

    def sample_action(self) -> int:
        return self.rng.integers(0, self.n_actions)
