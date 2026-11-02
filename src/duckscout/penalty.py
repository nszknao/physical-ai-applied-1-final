"""Goal-line evaluation for the axis-aligned penalty scene; never steers the robot."""

from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass(kw_only=True)
class GoalMonitor:
    model: mujoco.MjModel
    ball_name: str = "ball"
    ball_qpos: int = field(init=False)
    radius: float = field(init=False)
    plane_x: float = field(init=False)
    y_min: float = field(init=False)
    y_max: float = field(init=False)
    height: float = field(init=False)
    scored: bool = field(init=False, default=False)
    _previous: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        # Named static goal geometry defines the opening. This evaluator assumes +X
        # is into the goal, as in penalty.xml; object positions are not policy inputs.
        self.ball_qpos = int(self.model.joint(f"{self.ball_name}_freejoint").qposadr[0])
        self.radius = float(self.model.geom(self.ball_name).size[0])
        left, right = self.model.geom("goal_left"), self.model.geom("goal_right")
        bar = self.model.geom("goal_crossbar")
        self.plane_x = float(self.model.site("goal_line").pos[0]) + self.radius
        self.y_min = float(right.pos[1] + right.size[0]) + self.radius
        self.y_max = float(left.pos[1] - left.size[0]) - self.radius
        self.height = float(bar.pos[2] - bar.size[0]) - self.radius

    def reset(self) -> None:
        self.scored = False
        self._previous = None

    def update(self, data: mujoco.MjData) -> bool:
        """Latch only a forward crossing with the entire ball inside the opening."""
        ball = data.qpos[self.ball_qpos : self.ball_qpos + 3].copy()
        previous = self._previous
        self._previous = ball
        if not self.scored and previous is not None:
            if previous[0] < self.plane_x <= ball[0]:
                # Interpolate the crossing so a fast ball cannot skip the opening.
                fraction = (self.plane_x - previous[0]) / (ball[0] - previous[0])
                crossing = previous + fraction * (ball - previous)
                self.scored = bool(
                    self.y_min <= crossing[1] <= self.y_max
                    and 0 <= crossing[2] <= self.height
                )
        return self.scored
