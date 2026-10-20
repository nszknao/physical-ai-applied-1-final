"""Tests for task-independent, streaming run records."""

import json
import tempfile
import unittest
from pathlib import Path

from duckscout.runlog import RunLog


class RunLogTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "run"

    def records(self):
        return [
            json.loads(line)
            for line in (self.output / "records.jsonl").read_text().splitlines()
        ]

    def test_arbitrary_states_and_time_reset_are_written_immediately(self):
        with RunLog(output=self.output, metadata={"code_revision": "abc"}) as log:
            log.start({"mode": "example"})
            log.write("event", 10.0, event="state_changed", state="ARRIVED")
            log.write("event", 10.0, event="state_changed", state="CUSTOM_STATE")
            log.write("event", 10.0, event="reset")
            log.write("sample", 0.0, position=[1, 2, 3])
            self.assertEqual(len(self.records()), 4)  # 終了を待たずにファイルへ届く。
        rows = self.records()
        self.assertEqual(rows[1]["state"], "CUSTOM_STATE")
        self.assertEqual(rows[3]["time_s"], 0)
        self.assertEqual(rows[-1]["reason"], "closed")
        self.assertEqual(
            json.loads((self.output / "config.json").read_text())["mode"], "example"
        )
        self.assertFalse((self.output / "result.json").exists())

    def test_exception_and_interrupt_are_logged_and_propagated(self):
        for exception, reason in (
            (RuntimeError("test error"), "error"),
            (KeyboardInterrupt(), "interrupted"),
        ):
            with self.subTest(reason=reason):
                output = self.output / reason
                with self.assertRaises(type(exception)):
                    with RunLog(output=output, metadata={}) as log:
                        log.start({})
                        log.write("sample", 1.0, value=2)
                        raise exception
                rows = [
                    json.loads(s)
                    for s in (output / "records.jsonl").read_text().splitlines()
                ]
                self.assertEqual(rows[-1]["reason"], reason)
                self.assertEqual(rows[-1]["error"], str(exception))
                self.assertTrue(log.stream.closed)

    def test_existing_directory_is_not_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "records.jsonl"
        sentinel.write_text("existing")
        with self.assertRaises(FileExistsError):
            with RunLog(output=self.output, metadata={}):
                self.fail("Existing output accepted")
        self.assertEqual(sentinel.read_text(), "existing")

    def test_invalid_sample_does_not_leave_partial_json(self):
        with self.assertRaises(ValueError):
            with RunLog(output=self.output, metadata={}) as log:
                log.start({})
                log.write("sample", 0.0, value=1)
                log.write("sample", 1.0, value=float("nan"))
        self.assertEqual(len(self.records()), 2)
        self.assertEqual(self.records()[-1]["reason"], "error")


if __name__ == "__main__":
    unittest.main()
