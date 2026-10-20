"""Tests for duckscout.__main__: CLI validation and process launch routing."""

import json
import os
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from duckscout.__main__ import ensure_viewer_process, main


class LauncherTests(unittest.TestCase):
    def test_non_macos_does_not_relaunch(self):
        with (
            patch.object(sys, "platform", "linux"),
            patch.object(os, "execve") as execute,
        ):
            ensure_viewer_process([])
        execute.assert_not_called()

    def test_restarted_process_consumes_marker_without_relaunching(self):
        with patch.object(sys, "platform", "darwin"):
            with patch.dict(os.environ, {"_DUCKSCOUT_MJPYTHON_STARTED": "1"}):
                with patch.object(os, "execve") as execute:
                    ensure_viewer_process([])
                self.assertNotIn("_DUCKSCOUT_MJPYTHON_STARTED", os.environ)
        execute.assert_not_called()

    def test_macos_preserves_interpreter_arguments_and_library_path(self):
        arguments = ["--scene", "/tmp/a scene.xml", "--keyframe", "SIT"]
        with patch.dict(os.environ, {"DYLD_FALLBACK_LIBRARY_PATH": "/existing/lib"}):
            original = os.environ.copy()
            with patch.object(sys, "platform", "darwin"):
                with patch.object(Path, "is_file", return_value=True):
                    with patch.object(os, "execve") as execute:
                        ensure_viewer_process(arguments)
            self.assertEqual(dict(os.environ), original)
        executable, command, environment = execute.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[2:], ["-m", "duckscout", *arguments])
        self.assertEqual(environment["_DUCKSCOUT_MJPYTHON_STARTED"], "1")
        self.assertEqual(
            environment["DYLD_FALLBACK_LIBRARY_PATH"],
            sysconfig.get_config_var("LIBDIR") + ":/existing/lib",
        )

    def test_missing_launcher_reports_setup_error(self):
        with patch.object(sys, "platform", "darwin"):
            with patch.object(Path, "is_file", return_value=False):
                with patch.object(os, "execve") as execute:
                    with self.assertRaisesRegex(RuntimeError, "mjpython is missing"):
                        ensure_viewer_process([])
        execute.assert_not_called()

    def test_headless_entry_does_not_launch_gui(self):
        with patch(
            "duckscout.__main__.ensure_viewer_process",
            side_effect=AssertionError("GUI launched"),
        ):
            self.assertEqual(main(["--headless", "--steps", "2"]), 0)


class ArgumentTests(unittest.TestCase):
    def test_auto_requires_policy_and_target_coordinates_must_be_finite(self):
        for arguments in (["--auto"], ["--target", "nan", "0"]):
            with self.subTest(arguments=arguments):
                with patch("duckscout.__main__.Simulation.from_scene") as load:
                    with self.assertRaises(SystemExit):
                        main(arguments)
                load.assert_not_called()

    def test_output_without_policy_or_camera_records_run_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch(
                "duckscout.simulation.mujoco.Renderer",
                side_effect=AssertionError("Graphics created"),
            ):
                self.assertEqual(
                    main(["--headless", "--steps", "2", "--output", str(output)]), 0
                )
            records = [
                json.loads(line)
                for line in (output / "records.jsonl").read_text().splitlines()
            ]
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(records[-2]["event"], "step_limit")
            self.assertEqual(records[-1]["reason"], "closed")
            self.assertIsNone(records[1]["command"])
            self.assertIn("runlog.py", " ".join(config["python_sources_sha256"]))
            self.assertIn("onnxruntime", config["versions"])
            self.assertEqual(len(config["inputs_sha256"]), 3)
            self.assertFalse((output / "result.json").exists())

    def test_rejects_wrong_extension_before_starting_simulation(self):
        with patch("duckscout.__main__.Simulation.from_scene") as load:
            with self.assertRaises(SystemExit):
                main(["--headless", "--snapshot", "frame.png"])
        load.assert_not_called()

    def test_plain_headless_does_not_create_graphics_context(self):
        with patch(
            "duckscout.simulation.mujoco.Renderer",
            side_effect=AssertionError("Graphics created"),
        ):
            self.assertEqual(main(["--headless", "--steps", "2"]), 0)


if __name__ == "__main__":
    unittest.main()
