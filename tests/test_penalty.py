"""Goal crossing geometry, independent of rendering, policies and inference."""

import unittest

import mujoco

from duckscout.penalty import GoalMonitor

SCENE = """<mujoco><worldbody>
<body><freejoint name="ball_freejoint"/>
<geom name="ball" type="sphere" size=".04"/></body>
<geom name="goal_left" type="capsule" fromto="1 .315 0 1 .315 .255" size=".015"/>
<geom name="goal_right" type="capsule" fromto="1 -.315 0 1 -.315 .255" size=".015"/>
<geom name="goal_crossbar" type="capsule" fromto="1 -.315 .255 1 .315 .255" size=".015"/>
<site name="goal_line" pos="1 0 0"/>
</worldbody></mujoco>"""


class GoalMonitorTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(SCENE)
        self.data = mujoco.MjData(self.model)
        self.goal = GoalMonitor(model=self.model)

    def move(self, x, y=0, z=0.04):
        self.data.qpos[:3] = x, y, z
        return self.goal.update(self.data)

    def test_entire_ball_must_cross_and_result_latches_until_reset(self):
        self.assertFalse(self.move(0.9))
        self.assertFalse(self.move(1.02))
        self.assertTrue(self.move(1.05))
        self.assertTrue(self.move(0.9))
        self.goal.reset()
        self.assertFalse(self.move(1.1))  # Spawning behind the line is not a goal.
        self.assertFalse(self.move(0.9))  # Backward crossing is not a goal.
        self.assertTrue(self.move(1.05))

    def test_outside_posts_or_above_bar_does_not_score(self):
        for y, z in ((0.27, 0.04), (-0.27, 0.04), (0, 0.21)):
            with self.subTest(y=y, z=z):
                self.goal.reset()
                self.assertFalse(self.move(0.9, y, z))
                self.assertFalse(self.move(1.1, y, z))

    def test_fast_crossing_is_evaluated_at_the_line(self):
        self.assertFalse(self.move(1.0, 0.1))
        self.assertTrue(self.move(1.1, 0.4))  # At x=1.04, y=.22 is inside.
