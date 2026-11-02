"""Counterfactual selection fixtures and results; no GPU required."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "check_selection", SCRIPTS / "check_selection.py"
)
check = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC.loader.exec_module(check)


class SelectionCasesTests(unittest.TestCase):
    def test_swapped_and_absent_cases_reset_both_free_balls(self):
        sim = Mock(key_id=0)
        sim.model.key_qpos = np.zeros((1, 35))
        sim.model.joint.side_effect = lambda name: Mock(
            qposadr=[21 if name.startswith("red") else 28]
        )
        placements = []
        sim.reset.side_effect = lambda: placements.append(sim.model.key_qpos[0].copy())
        with (
            patch.object(check.Simulation, "from_scene", return_value=sim),
            patch.object(check.mujoco, "Renderer"),
            patch.object(check, "render_head_camera"),
            patch.object(check, "save_snapshot"),
        ):
            paths = check.capture_cases(Path("frames"))
        self.assertEqual([p.stem for p in paths], list(check.CASES))
        for actual, (red, blue, _) in zip(
            placements, check.CASES.values(), strict=True
        ):
            np.testing.assert_allclose(actual[21:24], red)
            np.testing.assert_allclose(actual[28:31], blue)
        self.assertEqual(sim.step.call_count, 1600)

    def test_wrong_color_and_invalid_json_cannot_count_as_success(self):
        report = {
            "images": [
                {
                    "image": "red_left.ppm",
                    "target_color": "red",
                    "validation_error": None,
                    "observation": {
                        "ball_position": "left",
                        "goal_position": "center",
                        "ball_inside_image": True,
                    },
                    "action": "turn_left",
                },
                {
                    "image": "red_left.ppm",
                    "target_color": "blue",
                    "validation_error": None,
                    "observation": {
                        "ball_position": "left",
                        "goal_position": "center",
                        "ball_inside_image": True,
                    },
                    "action": "turn_left",
                },
                {
                    "image": "red_absent.ppm",
                    "target_color": "red",
                    "validation_error": "bad JSON",
                    "observation": None,
                    "action": "stop",
                },
            ]
        }
        check.evaluate(report)
        self.assertEqual([r["passed"] for r in report["images"]], [True, False, False])
        self.assertEqual(report["evaluation"]["passed"], 1)

    def test_reused_frames_do_not_capture_again_and_output_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"

            def fake_analysis(*args, **kwargs):
                args[1].mkdir()
                return {"images": []}

            with (
                patch.object(check, "capture_cases") as capture,
                patch.object(
                    check, "analyze_images", side_effect=fake_analysis
                ) as analyze,
            ):
                self.assertEqual(
                    check.main(["--frames", "frames", "--output", str(output)]), 0
                )
                self.assertEqual(
                    check.main(["--frames", "frames", "--output", str(output)]), 1
                )
            capture.assert_not_called()
            self.assertEqual(analyze.call_args.kwargs["target_colors"], ("red", "blue"))
            self.assertEqual(len(analyze.call_args.args[0]), 4)
