"""One visual observation and bounded motion; physics stays on the caller thread."""

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from duckscout.vision import VisionModel

# Simulation seconds: wait for balance, move briefly, then leave the command zero.
SETTLE_SECONDS = 0.5
INITIAL_STAND_SECONDS = 2.0
MOVE_SECONDS = 1.0
# N toggles MuJoCo Island coloring; digits 0-5 toggle geometry groups.
STEP_KEY = ord("8")
COMMANDS = {
    "forward": (0.30, 0.0),  # m/s, rad/s
    "turn_left": (0.0, 0.8),
    "turn_right": (0.0, -0.8),
    "stop": (0.0, 0.0),
}


def choose_action(observation: dict | None) -> str:
    """Map image directions to robot directions, without VLM-generated actions."""
    if observation is None or observation.get("ball_fully_visible") is not True:
        return "stop"
    return {"left": "turn_left", "center": "forward", "right": "turn_right"}.get(
        observation.get("ball_position"), "stop"
    )


@dataclass(kw_only=True)
class VisualStep:
    """One worker owns all MLX calls. Main-thread methods never wait on inference."""

    loader: Callable[[], VisionModel] = VisionModel.load
    state: str = field(init=False, default="LOADING")
    action: str = field(init=False, default="stop")
    result: dict | None = field(init=False, default=None)
    error: str | None = field(init=False, default=None)
    observed_at: float | None = field(init=False, default=None)
    _executor: ThreadPoolExecutor = field(init=False, repr=False)
    _future: Future | None = field(init=False, default=None, repr=False)
    _model: VisionModel | None = field(init=False, default=None, repr=False)
    _deadline: float = field(init=False, default=0.0)
    _discard: bool = field(init=False, default=False)
    _started: float = field(init=False, default_factory=time.monotonic)

    def __enter__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision")
        # Import/load/generate stay on this single worker (MLX streams are thread-local).
        self._future = self._executor.submit(self.loader)
        return self

    def __exit__(self, *_):
        self.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def request(self, sim_time: float) -> bool:
        if self.state != "READY" or self._future is not None:
            return False  # No request queue, even if the step key is held down.
        self.action = "stop"
        self.error = None
        self.result = None
        self.state = "SETTLING"
        self._deadline = max(sim_time + SETTLE_SECONDS, INITIAL_STAND_SECONDS)
        return True

    def needs_image(self, sim_time: float) -> bool:
        return self.state == "SETTLING" and sim_time + 1e-9 >= self._deadline

    def observe(self, rgb: np.ndarray, sim_time: float) -> None:
        if not self.needs_image(sim_time):
            raise RuntimeError("A fresh image is only accepted after settling")
        self.observed_at = float(sim_time)
        self._started = time.monotonic()
        self._discard = False
        self.state = "THINKING"
        self._future = self._executor.submit(self._model.analyze, rgb.copy())

    def cancel(self) -> None:
        self.action = "stop"
        self._discard = True
        self.result = None
        self.observed_at = None
        if self.state == "LOADING":
            return  # Keep the model load; it does not contain an observation.
        if self._future is not None:
            self.state = "CANCELLING"
        elif self.state != "ERROR":
            self.state = "READY"

    def update(self, sim_time: float) -> tuple[float, float] | None:
        """Poll before each physics step; return only command changes to apply."""
        if self._future is not None and self._future.done():
            future, self._future = self._future, None
            if self.state != "LOADING" and self._discard:
                self.state = "READY"
                return None  # Discard without overwriting a later manual command.
            try:
                value = future.result()  # Already done; never blocks the physics loop.
                if self.state == "LOADING":
                    self._model = value
                    self.state = "READY"
                    return None
                self.result = value
                observation = (
                    None if value.get("validation_error") else value.get("observation")
                )
                self.action = choose_action(observation)
                self.result = {**value, "action": self.action}
                self.state = "READY" if self.action == "stop" else "MOVING"
                self._deadline = sim_time + MOVE_SECONDS
                return COMMANDS[self.action]
            except Exception as error:
                # Background failures must never leave a nonzero motion command.
                self.error = str(error)
                self.state = "ERROR"
                self.action = "stop"
                return COMMANDS["stop"]
        if self.state == "MOVING" and sim_time + 1e-9 >= self._deadline:
            self.state = "READY"
            self.action = "stop"
            return COMMANDS["stop"]
        return None

    def status_lines(self, paused: bool) -> list[str]:
        state = self.state
        if state in ("LOADING", "THINKING", "CANCELLING"):
            state += f" {time.monotonic() - self._started:.1f}s"
        lines = [
            f"VLM {state}",
            "Space: run | 8: one step" if paused else "8: one step | Down: cancel",
        ]
        if self.result is not None:
            obs = self.result.get("observation") or {}
            lines.append(
                f"ball: {obs.get('ball_position', '?')}  goal: {obs.get('goal_position', '?')}"
            )
            lines.append(
                f"infer {self.result['inference_seconds']:.2f}s | last {self.result['action']}"
            )
            if self.result.get("validation_error"):
                lines[-1] = "Invalid answer: stopped"
        if self.error is not None:
            lines[-1] = "Error: see terminal"
        return lines
