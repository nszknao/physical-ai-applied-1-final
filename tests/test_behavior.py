"""Tests for duckscout.behavior using only detections, estimated range and time."""

import unittest

from duckscout.behavior import Behavior, State
from duckscout.locomotion import FORWARD_SPEED, TURN_SPEED
from duckscout.perception import Detection


def target(cx=160):
    return Detection(bbox=(int(cx) - 15, 90, 30, 30), center=(cx, 105.0), area=900)


class BehaviorTests(unittest.TestCase):
    def setUp(self):
        self.behavior = Behavior()

    def acquire(self, cx=160, distance=0.8):
        self.assertEqual(self.behavior.update(target(cx), distance, 1.0), (0, 0))
        return self.behavior.update(target(cx), distance, 1.2)

    def test_settles_then_searches_without_forward_motion(self):
        self.assertEqual(self.behavior.update(None, None, 0.9), (0, 0))
        self.assertEqual(self.behavior.update(None, None, 1.0), (0, TURN_SPEED))
        self.assertEqual(self.behavior.state, State.SEARCH)

    def test_requires_sustained_detection_and_ignores_same_time_repeats(self):
        self.behavior.update(target(), 0.8, 1.0)
        for _ in range(10):
            self.assertEqual(self.behavior.update(target(), 0.8, 1.0), (0, 0))
        self.behavior.update(None, None, 1.1)
        self.assertEqual(self.behavior.update(target(), 0.8, 1.2), (0, 0))
        self.assertEqual(self.behavior.update(target(), 0.8, 1.4), (FORWARD_SPEED, 0))

    def test_turns_toward_target_then_walks_with_hysteresis(self):
        self.assertEqual(self.acquire(cx=100), (0, TURN_SPEED))
        self.assertEqual(self.behavior.state, State.ALIGN)
        self.assertEqual(
            self.behavior.update(target(175), 0.8, 1.3), (FORWARD_SPEED, 0)
        )
        self.assertEqual(
            self.behavior.update(target(190), 0.8, 1.4), (FORWARD_SPEED, 0)
        )
        self.assertEqual(self.behavior.update(target(210), 0.8, 1.5), (0, -TURN_SPEED))
        self.assertEqual(self.behavior.update(target(190), 0.8, 1.6), (0, -TURN_SPEED))
        self.assertEqual(
            self.behavior.update(target(160), 0.8, 1.7), (FORWARD_SPEED, 0)
        )

    def test_loss_stops_immediately_and_waits_before_searching(self):
        self.acquire()
        self.assertEqual(self.behavior.update(None, None, 1.3), (0, 0))
        self.assertEqual(self.behavior.update(None, None, 1.7), (0, 0))
        self.assertEqual(self.behavior.update(None, None, 1.9), (0, TURN_SPEED))
        self.assertEqual(self.behavior.state, State.SEARCH)
        self.assertEqual(self.behavior.update(target(), 0.8, 2.0), (0, 0))
        self.assertEqual(self.behavior.update(target(), 0.8, 2.2), (FORWARD_SPEED, 0))

    def test_invalid_or_clipped_distance_stops_without_claiming_arrival(self):
        self.acquire()
        for i, distance in enumerate((None, float("nan"), float("inf"), -1, 0)):
            self.assertEqual(
                self.behavior.update(target(), distance, 1.3 + i / 10), (0, 0)
            )
            self.assertNotEqual(self.behavior.state, State.ARRIVED)

    def test_arrival_requires_confirmation_then_stays_stopped(self):
        self.acquire()
        self.assertEqual(self.behavior.update(target(), 0.39, 1.3), (0, 0))
        self.assertNotEqual(self.behavior.state, State.ARRIVED)
        self.assertEqual(self.behavior.update(target(), 0.42, 1.4), (0, 0))
        self.assertEqual(self.behavior.update(target(), 0.40, 1.5), (0, 0))
        self.assertEqual(self.behavior.state, State.ARRIVED)
        self.assertEqual(self.behavior.update(None, None, 2.0), (0, 0))
        self.assertEqual(self.behavior.update(target(20), 1.0, 90.0), (0, 0))
        self.assertEqual(self.behavior.state, State.ARRIVED)

    def test_single_near_frame_does_not_latch_arrival(self):
        self.acquire()
        self.behavior.update(target(), 0.3, 1.3)
        self.assertEqual(self.behavior.update(target(), 0.8, 1.4), (FORWARD_SPEED, 0))
        self.assertNotEqual(self.behavior.state, State.ARRIVED)

    def test_timeout_latches_stop_even_if_target_later_appears(self):
        self.assertEqual(self.behavior.update(None, None, 60), (0, 0))
        self.assertEqual(self.behavior.state, State.TIMED_OUT)
        self.assertEqual(self.behavior.update(target(), 0.2, 61), (0, 0))
        self.assertEqual(self.behavior.state, State.TIMED_OUT)

    def test_cancel_and_reset_clear_state_and_time(self):
        self.acquire()
        self.behavior.cancel()
        self.assertEqual(self.behavior.update(target(), 0.8, 2), (0, 0))
        self.assertEqual(self.behavior.state, State.MANUAL)
        self.behavior.reset()
        self.assertEqual(self.behavior.state, State.SEARCH)
        self.assertIsNone(self.behavior.seen_since)
        self.assertEqual(self.behavior.update(None, None, 0), (0, 0))

    def test_reversed_or_nonfinite_time_is_rejected(self):
        self.acquire()
        for time in (0.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.behavior.update(target(), 0.8, time)

    def test_invalid_configuration_is_rejected(self):
        for args in (
            {"image_width": 0},
            {"stop_distance": -1},
            {"timeout": float("nan")},
            {"align_enter": 0.2, "align_exit": 0.1},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Behavior(**args)


if __name__ == "__main__":
    unittest.main()
