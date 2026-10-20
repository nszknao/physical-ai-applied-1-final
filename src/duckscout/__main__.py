"""Command-line entry point for the MicroDuck viewer."""

import argparse
import os
import sys
import sysconfig
from pathlib import Path

from duckscout.simulation import load_scene, run_headless, run_viewer


def ensure_viewer_process(arguments: list[str]) -> None:
    if sys.platform != "darwin":
        return

    # Consume our own restart marker so later child processes do not inherit it.
    if os.environ.pop("_DUCKSCOUT_MJPYTHON_STARTED", None) == "1":
        return

    launcher = Path(sysconfig.get_path("scripts")) / "mjpython"
    if not launcher.is_file():
        raise RuntimeError("mjpython is missing. Run uv sync first.")
    environment = os.environ.copy()
    library_dir = sysconfig.get_config_var("LIBDIR")
    library_name = sysconfig.get_config_var("LDLIBRARY")
    if library_dir and library_name and (Path(library_dir) / library_name).is_file():
        # uv's .venv/bin/python is a symlink; mjpython otherwise searches for
        # @executable_path-relative libpython inside the virtual environment.
        existing = environment.get(
            "DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib"
        )
        environment["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
            part for part in (library_dir, existing) if part
        )
    environment["_DUCKSCOUT_MJPYTHON_STARTED"] = "1"
    command = [sys.executable, str(launcher), "-m", "duckscout", *arguments]
    os.execve(sys.executable, command, environment)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MicroDuck model viewer")
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path(
            "assets/vendor/microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml"
        ),
        help="MJCF scene path (relative to the current directory)",
    )
    parser.add_argument("--keyframe", default="STAND", help="Initial named pose")
    parser.add_argument(
        "--headless", action="store_true", help="Check physics without a window"
    )
    parser.add_argument(
        "--steps", type=int, default=300, help="Number of headless physics steps"
    )
    arguments = sys.argv[1:] if arguments is None else arguments
    args = parser.parse_args(arguments)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not args.scene.is_file():
        parser.error(
            f"Scene not found: {args.scene}\n"
            "Run from the repository root after: python3.12 scripts/fetch_assets.py"
        )
    try:
        if not args.headless:
            ensure_viewer_process(arguments)
        model, data, key_id = load_scene(args.scene, args.keyframe)
        if args.headless:
            run_headless(model, data, args.steps)
        else:
            run_viewer(model, data, key_id)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f"duckscout: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
