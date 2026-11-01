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

MODEL_ID = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
MODEL_REVISION = "2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b"
MAX_TOKENS = 256
# This prompt describes the task, not the simulator's object coordinates.
PROMPT = """You see one camera image from a small walking robot on a green field.
The task is to approach an orange/red ball and push it into a white goal.
Describe only what is visible. Do not infer hidden objects or metric distances.
Return ONLY one JSON object with exactly these keys:
{"ball_position": "left|center|right|not_visible|unknown",
 "goal_position": "left|center|right|not_visible|unknown",
 "action": "forward|turn_left|turn_right|stop", "reason": "short explanation"}
Positions refer to the left, middle, or right third of the image.
Use the center of the goal opening for goal_position.
Choose ONE allowed value per field, not the list of alternatives.
Recommend turning toward the ball if it is to the left or right. Recommend
forward only if the ball and goal are both centered and clearly visible.
Otherwise recommend stop, including when the ball is cut off at the bottom
or the scene is uncertain. Explain the recommendation in one short sentence.
This is an image inspection, not a claim that the goal has been scored.
"""


def parse_decision(text: str) -> dict:
    """Accept plain JSON or a single Markdown JSON fence, then check its schema."""
    text = text.strip()
    if text.startswith("```json\n") or text.startswith("```\n"):
        lines = text.splitlines()
        if lines[-1] == "```":
            text = "\n".join(lines[1:-1])
    result = json.loads(text)
    fields = {"ball_position", "goal_position", "action", "reason"}
    if not isinstance(result, dict) or set(result) != fields:
        raise ValueError(
            "Expected exactly ball_position, goal_position, action, reason"
        )
    positions = ("left", "center", "right", "not_visible", "unknown")
    for field in ("ball_position", "goal_position"):
        if result[field] not in positions:
            raise ValueError(f"Invalid {field}: {result[field]!r}")
    if result["action"] not in ("forward", "turn_left", "turn_right", "stop"):
        raise ValueError(f"Invalid action: {result['action']!r}")
    if not isinstance(result["reason"], str) or not result["reason"].strip():
        raise ValueError("reason must be a nonempty string")
    return result


def prepare_image(rgb: np.ndarray) -> np.ndarray:
    """Validate camera RGB and bound image size while preserving aspect ratio."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("Expected a nonempty uint8 RGB image")
    height, width = rgb.shape[:2]
    scale = min(1.0, 640 / max(width, height))
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
        }
        try:
            record["decision"] = parse_decision(result.text)
            record["validation_error"] = None
        except ValueError as error:
            record["decision"] = None
            record["validation_error"] = str(error)
        return record
