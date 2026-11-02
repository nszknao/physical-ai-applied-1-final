"""Image analysis schema and routing tests; no downloads or Metal initialization."""

import importlib.util
import json
import unittest

import numpy as np

from duckscout.vision import (
    MODEL_ID,
    MODEL_REVISION,
    PROMPT,
    VisionModel,
    parse_observation,
    prepare_image,
)

ANSWER = {
    "ball_position": "center",
    "goal_position": "center",
    "ball_fully_visible": True,
}


class ObservationTests(unittest.TestCase):
    def test_plain_and_fenced_json(self):
        text = json.dumps(ANSWER)
        for raw in (text, f"```json\n{text}\n```", f"```\n{text}\n```"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_observation(raw), ANSWER)

    def test_rejects_incomplete_or_unstructured_responses(self):
        for raw in ("", "I would move forward", "{}", "[]", "null", '{"action":'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_observation(raw)

    def test_rejects_unknown_fields_enums_and_non_boolean_visibility(self):
        for replacement in (
            {"action": "kick"},
            {"ball_position": None},
            {"goal_position": []},
            {"ball_fully_visible": "true"},
            {"ball_fully_visible": 1},
            {"confidence": 1.0},
            {"ball_position": "not_visible", "ball_fully_visible": True},
        ):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                parse_observation(json.dumps(ANSWER | replacement))


class ImageInputTests(unittest.TestCase):
    def test_resize_preserves_rgb_and_aspect_ratio(self):
        rgb = np.zeros((480, 960, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        resized = prepare_image(rgb)
        self.assertEqual(resized.shape, (320, 640, 3))
        np.testing.assert_array_equal(resized[0, 0], [255, 0, 0])

    def test_invalid_camera_images_fail_before_backend_import(self):
        model = VisionModel(model=None, processor=None, prompt="test")
        for rgb in (
            np.zeros((3, 3)),
            np.zeros((3, 3, 4), np.uint8),
            np.zeros((0, 3, 3), np.uint8),
        ):
            with self.subTest(shape=rgb.shape), self.assertRaises(ValueError):
                model.analyze(rgb)


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
