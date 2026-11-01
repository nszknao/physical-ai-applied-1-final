"""Inspect saved camera images with a local VLM; never issue robot commands."""

import hashlib
import json
import platform
import sys
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import cv2

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


def analyze_images(paths: list[Path], output: Path | None = None) -> dict:
    """Load once, infer sequentially, and retain raw answers for human review."""
    if not paths:
        raise ValueError("At least one image is required")
    if output is not None and output.exists():
        raise FileExistsError(
            f"Output already exists: {output}; choose a new directory"
        )
    images = []
    records = []
    # Decode before loading/downloading the model so bad paths fail promptly.
    for path in paths:
        content = path.read_bytes()
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot decode image: {path}")
        height, width = frame.shape[:2]
        scale = min(1.0, 640 / max(width, height))
        if scale < 1:
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        records.append(
            {
                "image": str(path.resolve()),
                "sha256": hashlib.sha256(content).hexdigest(),
                "original_size": [width, height],
                "input_size": [frame.shape[1], frame.shape[0]],
            }
        )
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX image inference requires an Apple Silicon Mac")

    # Optional MLX imports initialize Metal. Keep them out of viewer/help/tests.
    try:
        from huggingface_hub import snapshot_download
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Install vision dependencies: uv sync --extra vision"
        ) from error
    except ImportError as error:
        raise RuntimeError(
            f"Cannot initialize MLX: {error} "
            "Run image analysis in a local terminal with Metal GPU access."
        ) from error

    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "prompt": PROMPT,
        "generation": {"max_tokens": MAX_TOKENS, "temperature": 0.0},
        "versions": {
            name: version(name) for name in ("mlx", "mlx-vlm", "transformers")
        },
        "images": [],
    }
    print(f"Loading {MODEL_ID} (first download: about 3 GB)...", file=sys.stderr)
    started = perf_counter()
    # The HF cache is keyed by the immutable revision; normal runs reuse it.
    with redirect_stdout(sys.stderr):
        model_path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, token=False)
        model, processor = load(model_path, trust_remote_code=False)
        config = json.loads((Path(model_path) / "config.json").read_text())
        prompt = apply_chat_template(processor, config, PROMPT, num_images=1)
    report["load_seconds"] = round(perf_counter() - started, 3)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    for index, (rgb, record) in enumerate(zip(images, records, strict=True)):
        print(f"Analyzing {record['image']}...", file=sys.stderr)
        started = perf_counter()
        with redirect_stdout(sys.stderr):
            result = generate(
                model,
                processor,
                prompt,
                image=[Image.fromarray(rgb)],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
                verbose=False,
            )
        record["inference_seconds"] = round(perf_counter() - started, 3)
        record["raw_response"] = result.text
        record["generation_tokens"] = result.generation_tokens
        try:
            record["decision"] = parse_decision(result.text)
            record["validation_error"] = None
        except ValueError as error:
            record["decision"] = None
            record["validation_error"] = str(error)
        report["images"].append(record)
        if output is not None:
            # Preserve the exact RGB input alongside the answer for review.
            Image.fromarray(rgb).save(output / f"input-{index + 1:02d}.png")
            (output / "result.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
    return report
