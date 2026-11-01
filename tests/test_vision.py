"""Image analysis schema and routing tests; no downloads or Metal initialization."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from duckscout.vision import (
    MODEL_ID,
    MODEL_REVISION,
    PROMPT,
    analyze_images,
    parse_decision,
)

ANSWER = {
    "ball_position": "center",
    "goal_position": "center",
    "action": "forward",
    "reason": "Both are centered.",
}


class DecisionTests(unittest.TestCase):
    def test_plain_and_fenced_json(self):
        text = json.dumps(ANSWER)
        for raw in (text, f"```json\n{text}\n```", f"```\n{text}\n```"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_decision(raw), ANSWER)

    def test_rejects_incomplete_or_unstructured_responses(self):
        for raw in ("", "I would move forward", "{}", "[]", "null", '{"action":'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_decision(raw)

    def test_rejects_unknown_fields_enums_and_empty_reason(self):
        for replacement in (
            {"action": "kick"},
            {"ball_position": None},
            {"goal_position": []},
            {"reason": "  "},
            {"reason": 42},
            {"confidence": 1.0},
        ):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                parse_decision(json.dumps(ANSWER | replacement))


class ImageAnalysisTests(unittest.TestCase):
    def test_bad_input_fails_before_optional_backend_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.png"
            path.write_text("not an image")
            with patch.dict(sys.modules, {"mlx_vlm": None}):
                with self.assertRaisesRegex(ValueError, "Cannot decode"):
                    analyze_images([path])
                with self.assertRaises(FileNotFoundError):
                    analyze_images([path.with_name("missing.png")])
                with self.assertRaises(FileExistsError):
                    analyze_images([path], Path(directory))

    def test_batch_loads_once_preserves_raw_answer_and_reports_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "camera.png"
            frame = np.zeros((480, 960, 3), dtype=np.uint8)
            frame[:, :, 2] = 255
            cv2.imwrite(str(image), frame)
            (root / "config.json").write_text('{"model_type": "qwen3_vl"}')
            generate = Mock(
                side_effect=[
                    SimpleNamespace(text=json.dumps(ANSWER), generation_tokens=50),
                    SimpleNamespace(text="truncated JSON {", generation_tokens=256),
                ]
            )
            load = Mock(return_value=("model", "processor"))
            snapshot = Mock(return_value=str(root))
            pil_image = Mock()
            pil_image.fromarray.return_value.save.side_effect = lambda path: (
                path.write_bytes(b"PNG stub")
            )
            backend_modules = {
                "huggingface_hub": SimpleNamespace(snapshot_download=snapshot),
                "mlx_vlm": SimpleNamespace(generate=generate, load=load),
                "mlx_vlm.prompt_utils": SimpleNamespace(
                    apply_chat_template=Mock(return_value="prompt")
                ),
                "PIL": SimpleNamespace(Image=pil_image),
            }
            with (
                patch.dict(sys.modules, backend_modules),
                patch("duckscout.vision.sys.platform", "darwin"),
                patch("duckscout.vision.platform.machine", return_value="arm64"),
                patch("duckscout.vision.version", return_value="test"),
            ):
                report = analyze_images([image, image], root / "output")
            load.assert_called_once_with(str(root), trust_remote_code=False)
            self.assertEqual(snapshot.call_args.kwargs["revision"], MODEL_REVISION)
            self.assertEqual(generate.call_count, 2)
            first, second = report["images"]
            self.assertEqual(first["decision"], ANSWER)
            self.assertEqual(first["original_size"], [960, 480])
            self.assertEqual(first["input_size"], [640, 320])
            self.assertEqual(len(first["sha256"]), 64)
            self.assertIsNone(second["decision"])
            self.assertIsNotNone(second["validation_error"])
            self.assertEqual(second["raw_response"], "truncated JSON {")
            self.assertEqual(
                json.loads((root / "output/result.json").read_text()), report
            )
            self.assertTrue((root / "output/input-02.png").exists())
            rgb = pil_image.fromarray.call_args.args[0]
            np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])


@unittest.skipUnless(
    importlib.util.find_spec("transformers") is not None,
    "Install the vision extra to test the real Qwen3-VL processor",
)
class QwenProcessorTests(unittest.TestCase):
    def test_cached_processor_accepts_a_still_image_without_metal(self):
        # Optional dependencies and cached model config; no network or MLX import.
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from PIL import Image
        from transformers import AutoProcessor

        try:
            model_path = snapshot_download(
                MODEL_ID, revision=MODEL_REVISION, local_files_only=True
            )
        except LocalEntryNotFoundError:
            self.skipTest("Qwen3-VL is not cached; run image analysis first")
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=False, local_files_only=True
        )
        prompt = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Match MLX-VLM's PyTorch fallback for Transformers fast processors.
        inputs = processor(
            text=[prompt], images=[Image.new("RGB", (320, 240))], return_tensors="pt"
        )
        self.assertEqual(inputs["input_ids"].shape[0], 1)
        self.assertEqual(inputs["image_grid_thw"].shape, (1, 3))
        self.assertEqual(
            inputs["pixel_values"].shape[0], inputs["image_grid_thw"].prod()
        )
        self.assertTrue(np.isfinite(inputs["pixel_values"].numpy()).all())


if __name__ == "__main__":
    unittest.main()
