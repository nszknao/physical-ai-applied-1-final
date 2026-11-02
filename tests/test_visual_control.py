"""Bounded visual actions, single-worker ownership and stale-result handling."""

import math
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from duckscout.visual_control import COMMANDS, MOVE_SECONDS, VisualStep, choose_action


def answer(position="center", fully_visible=True, error=None):
    return {
        "observation": {
            "ball_position": position,
            "goal_position": "center",
            "ball_inside_image": fully_visible,
        },
        "validation_error": error,
        "inference_seconds": 4.0,
        "raw_response": "test answer",
        "generation_tokens": 24,
    }


class ActionTests(unittest.TestCase):
    def test_image_left_means_left_and_unreliable_observations_stop(self):
        for position, expected in (
            ("left", "turn_left"),
            ("center", "forward"),
            ("right", "turn_right"),
            ("not_visible", "stop"),
            ("unknown", "stop"),
        ):
            self.assertEqual(choose_action(answer(position)["observation"]), expected)
        for value in (None, {}, answer("left", False)["observation"]):
            self.assertEqual(choose_action(value), "stop")
        self.assertEqual(
            choose_action(answer("left")["observation"] | {"action": "turn_right"}),
            "turn_left",
        )
        self.assertGreater(COMMANDS["turn_left"][1], 0)
        self.assertLess(COMMANDS["turn_right"][1], 0)


class _WorkerTestCase(unittest.TestCase):
    def setUp(self):
        self.release = threading.Event()
        self.started = threading.Event()
        self.worker_threads = []
        self.received = []
        self.requested_colors = []
        self.response = answer()

        def analyze(rgb, *, target_color=None):
            self.requested_colors.append(target_color)
            self.worker_threads.append(threading.get_ident())
            self.started.set()
            if not self.release.wait(3):
                raise TimeoutError("test worker was not released")
            self.received.append(rgb.copy())
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

        def load():
            self.worker_threads.append(threading.get_ident())
            return SimpleNamespace(analyze=analyze)

        self.control = VisualStep(loader=load).__enter__()
        self.addCleanup(self.close)
        self.control._future.result(timeout=3)
        self.assertIsNone(self.control.update(0))
        self.assertEqual(self.control.state, "READY")

    def close(self):
        self.release.set()
        self.control.__exit__(None, None, None)
        self.control._executor.shutdown(wait=True)

    def observe(self):
        self.assertTrue(self.control.request(0))
        self.assertFalse(self.control.needs_image(0.5))
        self.assertTrue(self.control.needs_image(2))
        image = np.full((2, 3, 3), 17, np.uint8)
        self.control.observe(image, 2)
        image[:] = 255  # Producer reuse cannot change the worker's observation.
        self.assertTrue(self.started.wait(3))

    def finish(self):
        self.release.set()
        try:
            self.control._future.result(timeout=3)
        except RuntimeError:
            pass  # The controller, rather than the test thread, handles this failure.


class VisualStepTests(_WorkerTestCase):
    def test_slow_inference_never_blocks_updates_and_motion_is_one_pulse(self):
        self.observe()
        for sim_time in (3, 4, 20):
            self.assertIsNone(self.control.update(sim_time))
            self.assertEqual(self.control.state, "THINKING")
        self.assertFalse(self.control.request(20))
        self.finish()
        self.assertEqual(self.control.update(20), COMMANDS["forward"])
        self.assertIsNone(self.control.update(20 + MOVE_SECONDS - 0.005))
        self.assertEqual(self.control.update(20 + MOVE_SECONDS), COMMANDS["stop"])
        self.assertIsNone(self.control.update(100))
        self.assertEqual(self.control.state, "READY")
        self.assertEqual(len(self.worker_threads), 2)
        self.assertEqual(self.worker_threads[0], self.worker_threads[1])
        self.assertNotEqual(self.worker_threads[0], threading.get_ident())
        np.testing.assert_array_equal(self.received[0], 17)
        self.assertIn("last forward", self.control.status_lines(False)[-1])

    def test_cancel_discards_old_result_without_replacing_manual_command(self):
        self.observe()
        self.control.cancel()
        self.assertFalse(self.control.request(0))
        self.finish()
        self.assertIsNone(self.control.update(0))
        self.assertIsNone(self.control.result)
        self.assertEqual(self.control.state, "READY")
        self.assertTrue(self.control.request(0))

    def test_cancel_during_settling_or_moving_prevents_later_action(self):
        self.assertTrue(self.control.request(0))
        self.control.cancel()
        self.assertFalse(self.control.needs_image(3))
        self.observe()
        self.finish()
        self.control.update(2)
        self.control.cancel()
        self.assertIsNone(self.control.update(3))
        self.assertEqual(self.control.action, "stop")

    def test_invalid_answer_stops_even_when_it_contains_a_visible_ball(self):
        self.response = answer(error="invalid JSON")
        self.observe()
        self.finish()
        self.assertEqual(self.control.update(3), COMMANDS["stop"])
        self.assertEqual(self.control.state, "READY")

    def test_inference_error_stops_and_is_reported(self):
        self.response = RuntimeError("model failure")
        self.observe()
        self.finish()
        self.assertEqual(self.control.update(3), COMMANDS["stop"])
        self.assertEqual(self.control.state, "ERROR")
        self.assertEqual(self.control.error, "model failure")
        self.assertFalse(self.control.request(4))

    def test_cancelled_error_cannot_disable_manual_control(self):
        self.response = RuntimeError("obsolete failure")
        self.observe()
        self.control.cancel()
        self.finish()
        self.assertIsNone(self.control.update(0))
        self.assertIsNone(self.control.error)

    def test_load_failure_is_polled_without_raising_on_physics_thread(self):
        def fail():
            raise RuntimeError("Metal unavailable")

        with VisualStep(loader=fail) as control:
            with self.assertRaises(RuntimeError):
                control._future.result(timeout=3)
            self.assertEqual(control.update(0), COMMANDS["stop"])
            self.assertEqual(control.state, "ERROR")
            self.assertIn("Metal unavailable", control.error)


class SoccerTests(_WorkerTestCase):
    def setUp(self):
        super().setUp()
        self.control.automatic = True

    # Single-step-specific tests remain in VisualStepTests. Here the shared worker
    # helpers exercise the same asynchronous API for an entire attempt.
    def test_slow_inference_never_blocks_updates_and_motion_is_one_pulse(self):
        self.observe()
        self.assertIsNone(self.control.update(8))
        self.finish()
        self.assertEqual(self.control.update(8), COMMANDS["forward"])
        self.assertEqual(self.control.update(10), COMMANDS["stop"])
        self.assertEqual(self.control.state, "SETTLING")
        self.assertFalse(self.control.needs_image(10.49))
        self.assertTrue(self.control.needs_image(10.5))
        self.response = answer(fully_visible=False)
        self.response["observation"]["ball_at_bottom"] = True
        self.control.observe(np.zeros((2, 2, 3), np.uint8), 10.5)
        self.finish()
        self.assertEqual(self.control.update(15), COMMANDS["forward"])
        self.assertEqual(self.control.state, "PUSHING")
        self.assertIsNone(self.control.update(22.99))
        self.assertEqual(self.control.update(23), COMMANDS["stop"])
        self.assertEqual(self.control.state, "STOPPED")
        self.assertFalse(self.control.running)
        self.assertTrue(self.control.request(24))

    def test_invalid_answer_stops_even_when_it_contains_a_visible_ball(self):
        self.response = answer(error="invalid JSON")
        self.observe()
        self.finish()
        self.assertEqual(self.control.update(3), COMMANDS["stop"])
        self.assertEqual(self.control.state, "STOPPED")

    def test_unaligned_or_unseen_ball_cannot_start_a_blind_push(self):
        for observation in (
            answer()["observation"] | {"goal_position": "left"},
            answer(fully_visible=False)["observation"] | {"ball_at_bottom": True},
            answer("not_visible", False)["observation"],
        ):
            self.control.cancel()
            self.response = answer() | {"observation": observation}
            self.observe()
            self.finish()
            self.assertEqual(self.control.update(3), COMMANDS["stop"])
            self.assertEqual(self.control.state, "STOPPED")

    def test_initial_clipped_result_stops_but_8_can_request_a_fresh_observation(self):
        self.response = answer(fully_visible=False)
        self.response["observation"]["ball_at_bottom"] = True
        self.observe()
        self.finish()
        self.assertEqual(self.control.update(8), COMMANDS["stop"])
        self.assertEqual(self.control.state, "STOPPED")
        self.assertIn("Clipped before approach", self.control.reason)
        self.assertIsNone(self.control.update(9))  # No automatic retry.
        self.assertTrue(self.control.request(10))
        self.assertFalse(self.control.needs_image(10.49))
        self.response = answer()
        self.control.observe(np.zeros((2, 2, 3), np.uint8), 10.5)
        self.finish()
        self.assertEqual(self.control.update(15), COMMANDS["forward"])
        self.assertEqual(self.control.state, "MOVING")

    def test_goal_discards_a_pending_result_and_does_not_restart(self):
        self.observe()
        self.assertEqual(self.control.finish("GOAL", "scored"), COMMANDS["stop"])
        self.finish()
        self.assertIsNone(self.control.update(10))
        self.assertEqual(self.control.state, "GOAL")
        self.assertFalse(self.control.request(11))
        self.control.cancel()
        self.assertTrue(self.control.request(0))

    def test_attempt_is_bounded_when_the_ball_never_gets_closer(self):
        from duckscout.visual_control import APPROACH_SECONDS, MAX_OBSERVATIONS

        self.observe()
        for index in range(MAX_OBSERVATIONS):
            now = 10 + index * 4
            self.finish()
            self.control.update(now)
            self.control.update(now + APPROACH_SECONDS)
            if index < MAX_OBSERVATIONS - 1:
                self.control.observe(np.zeros((2, 2, 3), np.uint8), now + 3)
        self.assertEqual(self.control.state, "STOPPED")
        self.assertFalse(self.control.running)


class ApproachTests(_WorkerTestCase):
    def setUp(self):
        super().setUp()
        self.control.automatic = True
        self.control.target_color = "blue"
        self.control.stop_before_push = True

    def test_selected_color_reaches_worker_and_near_stops_without_pushing(self):
        self.response = answer(fully_visible=False)
        self.response["observation"]["ball_at_bottom"] = True
        self.observe()
        self.finish()
        self.assertEqual(self.control.update(8, heading=0), COMMANDS["stop"])
        self.assertEqual(self.control.state, "NEAR")
        self.assertEqual(self.requested_colors, ["blue"])
        self.assertFalse(self.control.running)
        self.assertIsNone(self.control.update(100, heading=0))
        self.assertFalse(self.control.request(101))

    def test_turn_ignores_initial_twist_and_stops_on_signed_heading_change(self):
        self.response = answer("left")
        self.observe()
        self.finish()
        self.assertEqual(
            self.control.update(8, heading=math.radians(179)), COMMANDS["turn_left"]
        )
        self.assertIsNone(self.control.update(8.5, heading=math.radians(-171)))
        self.assertIsNone(
            self.control.update(9.3, heading=math.radians(169))
        )  # Wrong direction.
        self.assertEqual(
            self.control.update(9.4, heading=math.radians(-171)), COMMANDS["stop"]
        )
        self.assertEqual(self.control.state, "SETTLING")

    def test_approach_does_not_require_goal_centering_but_still_requires_target(self):
        self.response = answer()
        self.response["observation"]["goal_position"] = "left"
        self.observe()
        self.finish()
        self.assertEqual(self.control.update(8, heading=0), COMMANDS["forward"])
        self.assertEqual(self.control.update(9, heading=0), COMMANDS["stop"])
        self.response = answer("not_visible", False)
        self.control.observe(np.zeros((2, 2, 3), np.uint8), 9.5)
        self.finish()
        self.assertEqual(self.control.update(15, heading=0), COMMANDS["stop"])
        self.assertEqual(self.control.state, "STOPPED")

    def test_selected_soccer_aligns_at_bottom_then_pushes_and_times_out(self):
        self.control.stop_before_push = False
        self.observe()
        self.finish()
        self.assertEqual(self.control.update(8, heading=0), COMMANDS["forward"])
        self.control.update(9, heading=0)
        self.response = answer(fully_visible=False)
        self.response["observation"].update(
            ball_at_bottom=True, tracking_position="left"
        )
        self.control.observe(np.zeros((2, 2, 3), np.uint8), 9.5)
        self.finish()
        self.assertEqual(self.control.update(14, heading=0), COMMANDS["turn_left"])
        self.control.update(15.2, heading=math.radians(10))
        self.response["observation"]["tracking_position"] = "center"
        self.control.observe(np.zeros((2, 2, 3), np.uint8), 15.7)
        self.finish()
        self.assertEqual(self.control.update(20, heading=0), COMMANDS["forward"])
        self.assertEqual(self.control.state, "PUSHING")
        self.assertEqual(self.control.update(28), COMMANDS["stop"])
        self.assertEqual(self.control.state, "STOPPED")

    def test_cancel_discards_selected_target_result(self):
        self.observe()
        self.control.cancel()
        self.finish()
        self.assertIsNone(self.control.update(10, heading=0))
        self.assertFalse(self.control.running)
        self.assertEqual(self.control.state, "READY")


if __name__ == "__main__":
    unittest.main()
