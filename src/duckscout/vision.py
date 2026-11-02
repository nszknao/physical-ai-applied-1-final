"""Local Qwen3-VL inference on RGB images, independent of files and simulation."""

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from duckscout.perception import detect_target

MODEL_ID = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
MODEL_REVISION = "2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b"
MAX_TOKENS = 32
IMAGE_MAX_EDGE = 320
# Named fields keep directions unambiguous; image boundaries are checked in pixels.
PROMPT = """Locate the orange/red ball and white goal in this robot camera image.
Return ONLY one JSON object with exactly these keys:
{"ball_position":"center","goal_position":"center"}
For EACH field choose ONE value: "left", "center", "right", "not_visible", "unknown".
Positions refer to the left, middle, or right horizontal third of the image.
Use the ball center and the center of the goal opening.
If an object is absent, use "not_visible". Report only visible evidence.
"""


def parse_observation(text: str) -> dict:
    """Validate the model's named directions, without accepting extra prose or lists."""
    text = text.strip()
    if text.startswith("```json\n") or text.startswith("```\n"):
        lines = text.splitlines()
        if lines[-1] == "```":
            text = "\n".join(lines[1:-1])
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != {
        "ball_position",
        "goal_position",
    }:
        raise ValueError("Expected exactly ball_position and goal_position")
    positions = ("left", "center", "right", "not_visible", "unknown")
    if any(value not in positions for value in result.values()):
        raise ValueError("Invalid ball or goal position")
    return result


def add_ball_boundary(rgb: np.ndarray, observation: dict) -> dict:
    """Check image borders for the red ball identified by the VLM.

    This penalty scene has a single orange/red ball. Reuse the color detector for
    pixel boundaries only; never replace the VLM's directions or missing objects.
    A red region in a different image third cannot validate the model's answer.
    """
    result = {**observation, "ball_inside_image": False, "ball_at_bottom": False}
    if observation["ball_position"] not in ("left", "center", "right"):
        return result
    detection = detect_target(rgb)
    if detection is None:
        return result
    height, width = rgb.shape[:2]
    third = min(2, int(detection.center[0] * 3 / width))
    if observation["ball_position"] != ("left", "center", "right")[third]:
        return result
    x, y, w, h = detection.bbox
    # Bbox right/bottom are exclusive; touching the last row means y+h == height.
    other_edges_clear = x > 0 and x + w < width and y > 0
    result["ball_inside_image"] = other_edges_clear and y + h < height
    result["ball_at_bottom"] = other_edges_clear and y + h == height
    return result


def prepare_image(rgb: np.ndarray) -> np.ndarray:
    """Validate camera RGB and bound image size while preserving aspect ratio."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("Expected a nonempty uint8 RGB image")
    height, width = rgb.shape[:2]
    scale = min(1.0, IMAGE_MAX_EDGE / max(width, height))
    if scale < 1:
        return cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return rgb


@dataclass
class VisionModel:
    """Loaded model and processor; construct with load(), reuse across frames."""

    model: Any
    processor: Any
    prompt: str

    @classmethod
    def load(cls) -> "VisionModel":
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("MLX image inference requires an Apple Silicon Mac")
        # MLX initializes Metal on import. Load optional dependencies only when
        # inference is requested, so simulation, CLI help and CPU tests still work.
        try:
            from huggingface_hub import snapshot_download
            from mlx_vlm import load
            from mlx_vlm.prompt_utils import apply_chat_template
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Install vision dependencies: uv sync --extra vision"
            ) from error
        except ImportError as error:
            raise RuntimeError(
                f"Cannot initialize MLX: {error} "
                "Run image analysis in a local terminal with Metal GPU access."
            ) from error
        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, token=False)
        model, processor = load(path, trust_remote_code=False)
        config = json.loads((Path(path) / "config.json").read_text())
        prompt = apply_chat_template(processor, config, PROMPT, num_images=1)
        return cls(model=model, processor=processor, prompt=prompt)

    def analyze(self, rgb: np.ndarray) -> dict:
        """Return the raw answer and schema validation; never issue commands."""
        rgb = prepare_image(rgb)
        # As in load(), keep optional Metal/Pillow dependencies at the API boundary.
        from mlx_vlm import generate
        from PIL import Image

        started = perf_counter()
        result = generate(
            self.model,
            self.processor,
            self.prompt,
            image=[Image.fromarray(rgb)],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            verbose=False,
        )
        record = {
            "inference_seconds": round(perf_counter() - started, 3),
            "raw_response": result.text,
            "generation_tokens": result.generation_tokens,
            "prompt_tokens": result.prompt_tokens,
            "prompt_tokens_per_second": result.prompt_tps,
            "generation_tokens_per_second": result.generation_tps,
            "mlx_peak_memory_gb": result.peak_memory,
        }
        try:
            record["observation"] = add_ball_boundary(
                rgb, parse_observation(result.text)
            )
            record["validation_error"] = None
        except ValueError as error:
            record["observation"] = None
            record["validation_error"] = str(error)
        return record
