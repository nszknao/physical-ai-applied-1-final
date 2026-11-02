"""Compare red/blue instructions on identical images and swapped ball placements."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import mujoco
from check_vision import analyze_images

from duckscout.simulation import Simulation, render_head_camera, save_snapshot

ROOT = Path(__file__).resolve().parents[1]
# Image-left is world +Y for this initial camera. Coordinates belong only to these
# verification fixtures; the model sees the saved image and a color instruction.
CASES = {
    "red_left": ((0.7, 0.2, 0.04), (0.7, -0.2, 0.04), {"red": "left", "blue": "right"}),
    "red_right": (
        (0.7, -0.2, 0.04),
        (0.7, 0.2, 0.04),
        {"red": "right", "blue": "left"},
    ),
    "red_absent": (
        (-0.7, 0, 0.04),
        (0.7, 0.2, 0.04),
        {"red": "not_visible", "blue": "left"},
    ),
    "blue_absent": (
        (0.7, 0.2, 0.04),
        (-0.7, 0, 0.04),
        {"red": "left", "blue": "not_visible"},
    ),
}


def capture_cases(output: Path) -> list[Path]:
    sim = Simulation.from_scene(
        ROOT / "assets/scenes/choice.xml",
        policy_path=ROOT / "assets/vendor/microduck-policies/velstand.onnx",
    )
    paths = []
    with mujoco.Renderer(sim.model, height=240, width=320) as renderer:
        for name, (red, blue, _) in CASES.items():
            for color, position in (("red", red), ("blue", blue)):
                qpos = int(sim.model.joint(f"{color}_ball_freejoint").qposadr[0])
                sim.model.key_qpos[sim.key_id, qpos : qpos + 3] = position
            sim.reset()
            for _ in range(400):
                sim.step()
            path = output / f"{name}.ppm"
            save_snapshot(path, render_head_camera(sim, renderer))
            paths.append(path)
    return paths


def evaluate(report: dict) -> None:
    """Compare declared fixture expectations after inference, never in its prompt."""
    for result in report["images"]:
        expected = CASES[Path(result["image"]).stem][2][result["target_color"]]
        observation = result.get("observation") or {}
        result["expected_ball_position"] = expected
        result["passed"] = (
            result.get("validation_error") is None
            and observation.get("ball_position") == expected
            and observation.get("goal_position") == "center"
            and observation.get("ball_inside_image") is (expected != "not_visible")
            and result.get("action")
            == {"left": "turn_left", "right": "turn_right", "not_visible": "stop"}[
                expected
            ]
        )
    report["evaluation"] = {
        "passed": sum(result["passed"] for result in report["images"]),
        "total": len(report["images"]),
        "scope": "Static image selection only; no movement is executed.",
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for images and results",
    )
    parser.add_argument(
        "--frames",
        type=Path,
        help="Reuse the four named PPM frames from an earlier check",
    )
    args = parser.parse_args(arguments)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        with redirect_stdout(sys.stderr):
            paths = (
                [args.frames / f"{name}.ppm" for name in CASES]
                if args.frames is not None
                else capture_cases(args.output / "frames")
            )
            # Load once. Each frame is passed unchanged with both instructions.
            report = analyze_images(
                paths, args.output / "analysis", target_colors=("red", "blue")
            )
        evaluate(report)
        (args.output / "analysis/result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(report["evaluation"]["passed"] != report["evaluation"]["total"])
    except (ImportError, OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"check_selection: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
