"""Measure actual motion from the configured VLM commands without a VLM or GUI."""

import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from duckscout.simulation import Simulation
from duckscout.visual_control import COMMANDS, MOVE_SECONDS

ROOT = Path(__file__).resolve().parents[1]


def yaw(quaternion) -> float:
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def measure_motion() -> list[dict]:
    results = []
    for action in ("forward", "turn_left", "turn_right"):
        simulation = Simulation.from_scene(
            ROOT / "assets/scenes/penalty.xml",
            policy_path=ROOT / "assets/vendor/microduck-policies/velstand.onnx",
        )
        timestep = simulation.model.opt.timestep
        for _ in range(round(2 / timestep)):
            simulation.step()
        start = simulation.data.qpos[:7].copy()
        forward, turn = COMMANDS[action]
        simulation.set_command(forward=forward, yaw_rate=turn)
        for _ in range(round(MOVE_SECONDS / timestep)):
            simulation.step()
        simulation.set_command()
        for _ in range(round(2 / timestep)):
            simulation.step()
        end = simulation.data.qpos[:7]
        heading_change = (yaw(end[3:7]) - yaw(start[3:7]) + math.pi) % (
            2 * math.pi
        ) - math.pi
        results.append(
            {
                "action": action,
                "command": {"forward_m_s": forward, "yaw_rad_s": turn},
                "command_seconds": MOVE_SECONDS,
                "settle_seconds": 2.0,
                "displacement_xy_m": (end[:2] - start[:2]).tolist(),
                "displacement_m": float(np.linalg.norm(end[:2] - start[:2])),
                "heading_change_degrees": math.degrees(heading_change),
                "min_height_m": simulation.min_height,
                "max_tilt_degrees": simulation.max_tilt,
            }
        )
    return results


def main() -> None:
    with redirect_stdout(sys.stderr):
        results = measure_motion()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
