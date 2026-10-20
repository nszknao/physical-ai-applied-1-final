"""Tests for duckscout.locomotion: policy observations, commands and contracts."""

import unittest

import mujoco
import numpy as np
from helpers import SCENE, fake_session, make_simulation

from duckscout.locomotion import (
    DEFAULT_POSE,
    FORWARD_SPEED,
    JOINT_NAMES,
    TURN_SPEED,
    LocomotionPolicy,
)


@unittest.skipUnless(
    SCENE.is_file(), "Run scripts/fetch_assets.py to install the scene"
)
class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.session = fake_session()
        self.sim = make_simulation(self.session)

    def test_commands_fill_twist_slots_and_preserve_action_history(self):
        self.sim.policy.last_action[:] = 0.2
        self.sim.set_command(forward=0.2, yaw_rate=0.5)
        observation = self.sim.policy.observation(self.sim.data)[0]
        np.testing.assert_allclose(observation[48:61], [0.2, 0, 0.5] + [0] * 10)
        np.testing.assert_allclose(observation[34:48], np.full(14, 0.2))
        self.sim.set_command()
        np.testing.assert_array_equal(self.sim.policy.command, np.zeros(13))
        np.testing.assert_allclose(self.sim.policy.last_action, np.full(14, 0.2))

    def test_invalid_commands_leave_previous_command_unchanged(self):
        self.sim.set_command(forward=0.2)
        before = self.sim.policy.command.copy()
        for forward, yaw in (
            (float("nan"), 0),
            (0, float("inf")),
            (-0.1, 0),
            (FORWARD_SPEED + 0.01, 0),
            (0, TURN_SPEED + 0.01),
            (0, -TURN_SPEED - 0.01),
        ):
            with self.subTest(forward=forward, yaw=yaw):
                with self.assertRaises(ValueError):
                    self.sim.set_command(forward=forward, yaw_rate=yaw)
                np.testing.assert_array_equal(self.sim.policy.command, before)

    def test_observation_order_frame_and_sensor_address(self):
        policy, data = self.sim.policy, self.sim.data
        root = policy.root_qpos
        # +90 degrees around x maps world down onto negative body y.
        data.qpos[root + 3 : root + 7] = [np.sqrt(0.5), np.sqrt(0.5), 0, 0]
        offsets = np.arange(14) * 0.01
        velocities = np.arange(14) * -0.02
        data.qpos[policy.qpos_indices] = DEFAULT_POSE + offsets
        data.qvel[policy.qvel_indices] = velocities
        mujoco.mj_forward(self.sim.model, data)
        data.sensordata[policy.gyro_address : policy.gyro_address + 3] = [1, 2, 3]
        policy.last_action[:] = np.arange(14) * 0.03
        obs = policy.observation(data)
        self.assertEqual(obs.shape, (1, 61))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs[0, :6], [1, 2, 3, 0, -1, 0], atol=1e-7)
        np.testing.assert_allclose(obs[0, 6:20], offsets, atol=1e-7)
        np.testing.assert_allclose(obs[0, 20:34], velocities)
        np.testing.assert_allclose(obs[0, 34:48], policy.last_action)
        np.testing.assert_array_equal(obs[0, 48:], np.zeros(13))

    def test_policy_contract_and_wrong_transmission_are_rejected(self):
        self.session.get_inputs.return_value[0].shape = [1, 51]
        with self.assertRaisesRegex(ValueError, "input"):
            LocomotionPolicy(model=self.sim.model, session=self.session)
        self.session.get_inputs.return_value[0].shape = [1, 61]
        self.sim.model.actuator_trnid[0, 0] = self.sim.model.actuator_trnid[1, 0]
        with self.assertRaisesRegex(ValueError, "actuator"):
            LocomotionPolicy(model=self.sim.model, session=self.session)

    def test_same_shape_policy_with_different_joint_order_is_rejected(self):
        metadata = self.session.get_modelmeta.return_value.custom_metadata_map
        metadata["joint_names"] = ",".join(reversed(JOINT_NAMES))
        with self.assertRaisesRegex(ValueError, "joint_names"):
            LocomotionPolicy(model=self.sim.model, session=self.session)


if __name__ == "__main__":
    unittest.main()
