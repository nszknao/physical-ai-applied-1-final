"""Offline inspection CLI, batch artifacts and camera-case capture checks."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from duckscout.vision import MODEL_REVISION

SPEC = importlib.util.spec_from_file_location(
    "check_vision", Path(__file__).resolve().parents[1] / "scripts/check_vision.py"
)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)

ANSWER = {
    "ball_position": "center",
    "goal_position": "center",
    "ball_inside_image": True,
    "ball_at_bottom": False,
}


class ImageAnalysisTests(unittest.TestCase):
    def test_bad_input_fails_before_optional_backend_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.png"
            path.write_text("not an image")
            with patch.dict(sys.modules, {"mlx_vlm": None}):
                with self.assertRaisesRegex(ValueError, "Cannot decode"):
                    check.analyze_images([path])
                with self.assertRaises(FileNotFoundError):
                    check.analyze_images([path.with_name("missing.png")])
                with self.assertRaises(FileExistsError):
                    check.analyze_images([path], Path(directory))

    def test_batch_loads_once_preserves_raw_answer_and_reports_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "camera.png"
            frame = np.zeros((480, 960, 3), dtype=np.uint8)
            frame[200:300, 450:510, 2] = 255
            cv2.imwrite(str(image), frame)
            (root / "config.json").write_text('{"model_type": "qwen3_vl"}')
            generate = Mock(
                side_effect=[
                    SimpleNamespace(
                        text=json.dumps(
                            {"ball_position": "center", "goal_position": "center"}
                        ),
                        generation_tokens=50,
                        prompt_tokens=100,
                        prompt_tps=20.0,
                        generation_tps=15.0,
                        peak_memory=3.5,
                    ),
                    SimpleNamespace(
                        text="truncated JSON {",
                        generation_tokens=96,
                        prompt_tokens=100,
                        prompt_tps=20.0,
                        generation_tps=15.0,
                        peak_memory=3.5,
                    ),
                ]
            )
            load = Mock(return_value=("model", "processor"))
            snapshot = Mock(return_value=str(root))
            pil_image = Mock()
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
                patch.object(check, "version", return_value="test"),
            ):
                report = check.analyze_images([image, image], root / "output")
            load.assert_called_once_with(str(root), trust_remote_code=False)
            self.assertEqual(snapshot.call_args.kwargs["revision"], MODEL_REVISION)
            self.assertEqual(generate.call_count, 2)
            first, second = report["images"]
            self.assertEqual(first["observation"], ANSWER)
            self.assertEqual(first["prompt_tokens_per_second"], 20.0)
            self.assertEqual(first["generation_tokens_per_second"], 15.0)
            self.assertEqual(first["mlx_peak_memory_gb"], 3.5)
            self.assertEqual(first["original_size"], [960, 480])
            self.assertEqual(first["input_size"], [320, 160])
            self.assertEqual(len(first["sha256"]), 64)
            self.assertIsNone(second["observation"])
            self.assertIsNotNone(second["validation_error"])
            self.assertEqual(second["raw_response"], "truncated JSON {")
            self.assertEqual(
                json.loads((root / "output/result.json").read_text()), report
            )
            self.assertTrue((root / "output/input-02.png").exists())
            rgb = pil_image.fromarray.call_args.args[0]
            np.testing.assert_array_equal(rgb[80, 160], [255, 0, 0])


class CheckCliTests(unittest.TestCase):
    def test_image_paths_bypass_capture_and_output_json(self):
        report = {"images": [{"validation_error": None}]}
        with (
            patch.object(check, "analyze_images", return_value=report) as analyze,
            patch.object(check, "capture_cases") as capture,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertEqual(check.main(["a.ppm", "b.png", "--output", "out"]), 0)
        analyze.assert_called_once_with([Path("a.ppm"), Path("b.png")], Path("out"))
        capture.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue()), report)

    def test_invalid_answer_returns_nonzero(self):
        with (
            patch.object(
                check,
                "analyze_images",
                return_value={"images": [{"validation_error": "bad JSON"}]},
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(check.main(["a.ppm"]), 1)

    def test_rejects_ambiguous_modes_and_missing_inputs(self):
        for args in ([], ["--cases"], ["a.ppm", "--cases", "--output", "out"]):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                check.main(args)

    def test_cases_capture_then_analyze_and_protect_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            paths = [output / "frames/center.ppm"]
            with (
                patch.object(check, "capture_cases", return_value=paths) as capture,
                patch.object(
                    check, "analyze_images", return_value={"images": []}
                ) as analyze,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(check.main(["--cases", "--output", str(output)]), 0)
                self.assertEqual(check.main(["--cases", "--output", str(output)]), 1)
            capture.assert_called_once_with(output / "frames")
            analyze.assert_called_once_with(paths, output / "analysis")


class CaptureTests(unittest.TestCase):
    def test_cases_reset_then_capture_without_passing_positions_to_inference(self):
        simulation = Mock()
        simulation.model.joint.return_value.qposadr = [21]
        simulation.model.key_qpos = np.zeros((1, 28))
        simulation.key_id = 0
        positions = []
        simulation.reset.side_effect = lambda: positions.append(
            simulation.model.key_qpos[0, 21:24].copy()
        )
        with (
            patch.object(check.Simulation, "from_scene", return_value=simulation),
            patch.object(check.mujoco, "Renderer"),
            patch.object(
                check,
                "render_head_camera",
                return_value=np.zeros((240, 320, 3), np.uint8),
            ) as render,
            patch.object(check, "save_snapshot") as save,
        ):
            paths = check.capture_cases(Path("frames"))
        self.assertEqual(
            [p.name for p in paths],
            ["center.ppm", "left.ppm", "right.ppm", "absent.ppm", "clipped.ppm"],
        )
        np.testing.assert_allclose(
            positions,
            [
                [0.7, 0, 0.04],
                [0.7, 0.2, 0.04],
                [0.7, -0.2, 0.04],
                [-0.7, 0, 0.04],
                [0.35, 0, 0.04],
            ],
        )
        self.assertEqual(simulation.step.call_count, 2000)
        self.assertEqual(render.call_count, 5)
        self.assertEqual(save.call_count, 5)


if __name__ == "__main__":
    unittest.main()
