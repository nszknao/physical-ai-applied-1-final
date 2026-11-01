"""Inspect saved images or four reproducible MuJoCo scenes with the real VLM."""

import argparse
import hashlib
import json
import sys
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import cv2
import mujoco

from duckscout.simulation import Simulation, render_head_camera, save_snapshot
from duckscout.vision import (
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT,
    VisionModel,
    prepare_image,
)

ROOT = Path(__file__).resolve().parents[1]


def capture_cases(output: Path) -> list[Path]:
    """Render centered/offset/hidden balls after two seconds of standing."""
    simulation = Simulation.from_scene(
        ROOT / "assets/scenes/penalty.xml",
        policy_path=ROOT / "assets/vendor/microduck-policies/velstand.onnx",
    )
    ball_qpos = int(simulation.model.joint("ball_freejoint").qposadr[0])
    paths = []
    with mujoco.Renderer(simulation.model, height=240, width=320) as renderer:
        for name, position in (
            ("center", (0.7, 0, 0.04)),
            ("left", (0.7, 0.2, 0.04)),
            ("right", (0.7, -0.2, 0.04)),
            ("absent", (-0.7, 0, 0.04)),
        ):
            # Fixture placement only: inference receives pixels, not these coordinates.
            simulation.model.key_qpos[simulation.key_id, ball_qpos : ball_qpos + 3] = (
                position
            )
            simulation.reset()
            for _ in range(400):
                simulation.step()
            path = output / f"{name}.ppm"
            save_snapshot(path, render_head_camera(simulation, renderer))
            paths.append(path)
    return paths


def analyze_images(paths: list[Path], output: Path | None = None) -> dict:
    """File input, batch timing and review artifacts belong to this check script."""
    if not paths:
        raise ValueError("At least one image is required")
    if output is not None and output.exists():
        raise FileExistsError(
            f"Output already exists: {output}; choose a new directory"
        )
    images = []
    records = []
    for path in paths:
        content = path.read_bytes()
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot decode image: {path}")
        height, width = frame.shape[:2]
        rgb = prepare_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        images.append(rgb)
        records.append(
            {
                "image": str(path.resolve()),
                "sha256": hashlib.sha256(content).hexdigest(),
                "original_size": [width, height],
                "input_size": [rgb.shape[1], rgb.shape[0]],
            }
        )
    print(f"Loading {MODEL_ID} (first download: about 3 GB)...", file=sys.stderr)
    started = perf_counter()
    with redirect_stdout(sys.stderr):
        vision = VisionModel.load()
    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "prompt": PROMPT,
        "generation": {"max_tokens": MAX_TOKENS, "temperature": 0.0},
        "versions": {
            name: version(name) for name in ("mlx", "mlx-vlm", "transformers")
        },
        "load_seconds": round(perf_counter() - started, 3),
        "images": [],
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    for index, (rgb, record) in enumerate(zip(images, records, strict=True)):
        print(f"Analyzing {record['image']}...", file=sys.stderr)
        with redirect_stdout(sys.stderr):
            record.update(vision.analyze(rgb))
        report["images"].append(record)
        if output is not None:
            path = output / f"input-{index + 1:02d}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Cannot save image: {path}")
            (output / "result.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
    return report


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("images", nargs="*", type=Path, metavar="IMAGE")
    parser.add_argument(
        "--cases", action="store_true", help="Capture and analyze four ball placements"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory for images and results; required with --cases",
    )
    args = parser.parse_args(arguments)
    if args.cases and (args.images or args.output is None):
        parser.error(
            "--cases requires --output and cannot be combined with image paths"
        )
    if not args.cases and not args.images:
        parser.error("Provide image paths or --cases")
    try:
        if args.cases:
            args.output.mkdir(parents=True, exist_ok=False)
            with redirect_stdout(sys.stderr):
                paths = capture_cases(args.output / "frames")
            report = analyze_images(paths, args.output / "analysis")
        else:
            report = analyze_images(args.images, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(any(item["validation_error"] for item in report["images"]))
    except (ImportError, OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"check_vision: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
