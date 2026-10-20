"""Image-driven search, alignment, approach and latched stop decisions."""

import math
from dataclasses import dataclass, field
from enum import StrEnum

from duckscout.locomotion import FORWARD_SPEED, TURN_SPEED
from duckscout.perception import Detection


class State(StrEnum):
    SEARCH = "SEARCH"
    ALIGN = "ALIGN"
    APPROACH = "APPROACH"
    ARRIVED = "ARRIVED"
    TIMED_OUT = "TIMED_OUT"
    MANUAL = "MANUAL"


@dataclass(kw_only=True)
class Behavior:
    """Configuration is injected; runtime state is reset internally."""

    image_width: int = 320
    stop_distance: float = 0.40  # Approximate optical-axis distance, metres.
    timeout: float = 60.0  # Simulation seconds, excluding pauses.
    settle_time: float = 1.0
    confirm_time: float = 0.2
    lost_grace: float = 0.5
    align_enter: float = 0.06  # Fraction of full image width from centre.
    align_exit: float = 0.12

    state: State = field(init=False, default=State.SEARCH)
    reason: str = field(init=False, default="settling")
    started_at: float = field(init=False, default=0.0)
    last_time: float = field(init=False, default=0.0)
    seen_since: float | None = field(init=False, default=None)
    missing_since: float | None = field(init=False, default=None)
    near_since: float | None = field(init=False, default=None)
    turn_direction: float = field(init=False, default=1.0)

    def __post_init__(self):
        if (
            self.image_width <= 0
            or not all(
                math.isfinite(value) and value > 0
                for value in (
                    self.stop_distance,
                    self.timeout,
                    self.settle_time,
                    self.confirm_time,
                    self.lost_grace,
                    self.align_enter,
                    self.align_exit,
                )
            )
            or not self.align_enter < self.align_exit < 0.5
        ):
            raise ValueError("Invalid behavior configuration")

    def reset(self, now: float = 0.0) -> None:
        self.state = State.SEARCH
        self.reason = "settling"
        self.started_at = self.last_time = now
        self.seen_since = self.missing_since = self.near_since = None
        self.turn_direction = 1.0

    def cancel(self) -> None:
        self.state = State.MANUAL
        self.reason = "manual control"

    def update(
        self, detection: Detection | None, distance: float | None, now: float
    ) -> tuple[float, float]:
        """Return (forward m/s, yaw rad/s) using only image evidence and time."""
        if not math.isfinite(now) or now < self.last_time:
            raise ValueError(
                "Behavior time must be finite and nondecreasing; reset first"
            )
        self.last_time = now
        if self.state in (State.ARRIVED, State.TIMED_OUT, State.MANUAL):
            return 0.0, 0.0
        if now - self.started_at >= self.timeout:
            self.state, self.reason = State.TIMED_OUT, "time limit"
            return 0.0, 0.0
        if now - self.started_at < self.settle_time:
            self.reason = "settling"
            return 0.0, 0.0
        if detection is None:
            self.seen_since = self.near_since = None
            if self.state != State.SEARCH:
                if self.missing_since is None:
                    self.missing_since = now
                if now - self.missing_since < self.lost_grace:
                    self.reason = "target lost; waiting"
                    return 0.0, 0.0
            self.state, self.reason = State.SEARCH, "searching"
            return 0.0, self.turn_direction * TURN_SPEED
        self.missing_since = None
        if self.seen_since is None:
            self.seen_since = now
        if now - self.seen_since + 1e-9 < self.confirm_time:
            self.reason = "confirming target"
            return 0.0, 0.0
        error = (detection.center[0] - self.image_width / 2) / self.image_width
        threshold = (
            self.align_exit if self.state == State.APPROACH else self.align_enter
        )
        if abs(error) > threshold:
            self.state, self.reason = State.ALIGN, "centering"
            self.near_since = None
            self.turn_direction = -1.0 if error > 0 else 1.0
            return 0.0, self.turn_direction * TURN_SPEED
        self.state = State.APPROACH
        if distance is None or not math.isfinite(distance) or distance <= 0:
            self.near_since = None
            self.reason = "distance unavailable; stopped"
            return 0.0, 0.0
        if distance <= self.stop_distance or (
            self.near_since is not None and distance <= self.stop_distance + 0.03
        ):
            if self.near_since is None:
                self.near_since = now
            self.reason = "confirming arrival"
            if now - self.near_since + 1e-9 >= self.confirm_time:
                self.state, self.reason = State.ARRIVED, "holding stand"
            return 0.0, 0.0
        self.near_since = None
        self.reason = "walking forward"
        return FORWARD_SPEED, 0.0
