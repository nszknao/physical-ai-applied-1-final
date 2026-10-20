"""Policy contracts and reset behavior against the downloaded MuJoCo scene."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np

from duckscout.locomotion import (
    DEFAULT_POSE,
    FORWARD_SPEED,
    JOINT_NAMES,
    PHYSICS_TIMESTEP,
    TURN_SPEED,
    LocomotionPolicy,
)
from duckscout.simulation import Simulation, configure_bam, handle_key

ROOT = Path(__file__).resolve().parents[1]
SCENE = (
    ROOT / "assets/vendor/microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml"
)
POLICY = ROOT / "assets/vendor/microduck-policies/velstand.onnx"


def fake_session():
    session = Mock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="obs", shape=[1, 61], type="tensor(float)")
    ]
    session.get_outputs.return_value = [
        SimpleNamespace(name="actions", shape=[1, 14], type="tensor(float)")
    ]
    session.get_modelmeta.return_value = SimpleNamespace(
        custom_metadata_map={
            "joint_names": ",".join(JOINT_NAMES),
            "observation_names": "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command",
            "command_names": "twist,head_pose,body_pose",
            "action_scale": "1.0",
        }
    )
    session.run.return_value = [np.zeros((1, 14), dtype=np.float32)]
    return session


@unittest.skipUnless(
    SCENE.is_file(), "Run scripts/fetch_assets.py to install the scene"
)
class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.session = fake_session()
        spec = mujoco.MjSpec.from_file(str(SCENE))
        bam = configure_bam(spec)
        spec.option.timestep = PHYSICS_TIMESTEP
        model = spec.compile()
        policy = LocomotionPolicy(model=model, session=self.session)
        self.sim = Simulation(
            model=model,
            data=mujoco.MjData(model),
            key_id=model.key("STAND").id,
            bam=bam,
            policy=policy,
        )

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

    def test_keys_switch_modes_stop_without_pausing_and_reset(self):
        paused = handle_key(self.sim, 265, True)  # Up queues motion while paused.
        self.assertTrue(paused)
        self.assertEqual(self.sim.data.time, 0)
        np.testing.assert_allclose(self.sim.policy.command[:3], [FORWARD_SPEED, 0, 0])
        paused = handle_key(self.sim, 32, paused)
        self.assertFalse(paused)
        self.sim.step()
        for key, yaw in ((263, TURN_SPEED), (262, -TURN_SPEED)):
            paused = handle_key(self.sim, key, paused)
            np.testing.assert_allclose(self.sim.policy.command[:3], [0, 0, yaw])
        paused = handle_key(self.sim, 264, paused)  # Down requests standing.
        self.assertFalse(paused)
        np.testing.assert_array_equal(self.sim.policy.command, np.zeros(13))
        before = self.sim.data.time
        self.sim.step()
        self.assertGreater(self.sim.data.time, before)
        self.assertTrue(handle_key(self.sim, 32, paused))
        self.sim.set_command(forward=FORWARD_SPEED)
        self.assertTrue(handle_key(self.sim, ord("R"), False))
        self.assertEqual(self.sim.data.time, 0)
        np.testing.assert_array_equal(self.sim.policy.command, np.zeros(13))
        np.testing.assert_array_equal(self.sim.policy.last_action, np.zeros(14))

    def test_commands_require_a_policy(self):
        sim = Simulation.from_scene(SCENE)
        with self.assertRaisesRegex(ValueError, "--policy"):
            sim.set_command(forward=FORWARD_SPEED)
        self.assertTrue(handle_key(sim, 265, True))

    def test_travel_is_allowed_by_posture_check(self):
        self.sim.data.qpos[self.sim.policy.root_qpos] = 1.0
        self.sim.check_upright()
        self.assertAlmostEqual(self.sim.max_displacement, 1.0)

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

    def test_history_is_raw_action_and_reset_clears_it(self):
        action = np.linspace(-0.2, 0.2, 14, dtype=np.float32)[None, :]
        self.session.run.return_value = [action]
        targets = self.sim.policy.targets(self.sim.data)
        np.testing.assert_allclose(targets, DEFAULT_POSE + action[0])
        self.sim.policy.targets(self.sim.data)
        feed = self.session.run.call_args.args[1]["obs"]
        np.testing.assert_array_equal(feed[0, 34:48], action[0])
        self.sim.reset()
        np.testing.assert_array_equal(self.sim.policy.last_action, np.zeros(14))
        np.testing.assert_array_equal(self.sim.motors.q_target, DEFAULT_POSE)

    def test_invalid_action_is_rejected_before_motor_target_changes(self):
        for action in (np.zeros((14,)), np.full((1, 14), np.nan)):
            with self.subTest(shape=action.shape):
                self.session.run.return_value = [action]
                before = self.sim.motors.q_target.copy()
                with self.assertRaisesRegex(RuntimeError, "Policy action"):
                    self.sim.step()
                np.testing.assert_array_equal(self.sim.motors.q_target, before)
                self.assertEqual(self.sim.data.time, 0)

    def test_nonfinite_observation_never_reaches_inference(self):
        self.sim.data.sensordata[self.sim.policy.gyro_address] = np.nan
        with self.assertRaisesRegex(RuntimeError, "observation"):
            self.sim.step()
        self.session.run.assert_not_called()

    def test_inference_is_50_hz_while_physics_is_200_hz(self):
        for _ in range(12):
            self.sim.step()
        self.assertEqual(self.session.run.call_count, 3)
        self.assertAlmostEqual(self.sim.data.time, 0.06)
        self.sim.reset()
        self.sim.step()
        self.assertEqual(self.session.run.call_count, 4)

    def test_reset_reproduces_trajectory_including_bam_friction(self):
        first = []
        for _ in range(40):
            self.sim.step()
            first.append(self.sim.data.qpos.copy())
        self.sim.reset()
        for expected in first:
            self.sim.step()
            np.testing.assert_allclose(self.sim.data.qpos, expected, atol=1e-12, rtol=0)

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

    def test_sitting_keyframe_is_rejected_for_standing(self):
        with self.assertRaisesRegex(ValueError, "STAND"):
            Simulation.from_scene(SCENE, keyframe="SIT", policy_path=Path("test.onnx"))

    def test_finite_but_fallen_robot_fails(self):
        root = self.sim.policy.root_qpos
        for position, quat in (
            ([0, 0, 0.05], [1, 0, 0, 0]),
            ([0, 0, 0.125], [np.sqrt(0.5), np.sqrt(0.5), 0, 0]),
        ):
            with self.subTest(position=position, quat=quat):
                self.sim.data.qpos[root : root + 7] = position + quat
                with self.assertRaisesRegex(RuntimeError, "Posture check failed"):
                    self.sim.check_upright()


def rollout(sim: Simulation, seconds: float) -> np.ndarray:
    """Sample x, y, unwrapped yaw and horizontal speed in simulation time."""
    root = sim.policy.root_qpos
    dof = int(sim.model.joint("trunk_base_freejoint").dofadr[0])
    samples = []
    steps = round(seconds / sim.model.opt.timestep)
    for index in range(steps + 1):
        if index:
            sim.step()
        x, y = sim.data.qpos[root : root + 2]
        qw, qx, qy, qz = sim.data.qpos[root + 3 : root + 7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        speed = np.linalg.norm(sim.data.qvel[dof : dof + 2])
        samples.append((x, y, yaw, speed))
    samples = np.array(samples)
    samples[:, 2] = np.unwrap(samples[:, 2])
    return samples


@unittest.skipUnless(
    SCENE.is_file() and POLICY.is_file(),
    "Run scripts/fetch_assets.py for real-policy checks",
)
class RealLocomotionTests(unittest.TestCase):
    def assert_stopped(self, sim, samples):
        # Allow 3 s to settle; evaluate the last 2 s of a 5 s stop interval.
        window = round(2 / sim.model.opt.timestep)
        tail = samples[-window - 1 :]
        mean_speed = np.mean(tail[1:, 3])
        mean_yaw_rate = np.mean(np.abs(np.diff(tail[:, 2]))) / sim.model.opt.timestep
        self.assertLess(mean_speed, 0.02)
        self.assertLess(mean_yaw_rate, 0.10)
        self.assertLess(np.max(np.linalg.norm(tail[:, :2] - tail[0, :2], axis=1)), 0.02)
        self.assertLess(np.ptp(tail[:, 2]), 0.10)
        return mean_speed, mean_yaw_rate

    def test_forward_left_right_then_stop(self):
        for label, forward, turn in (
            ("forward", FORWARD_SPEED, 0),
            ("left", 0, TURN_SPEED),
            ("right", 0, -TURN_SPEED),
        ):
            with self.subTest(motion=label):
                sim = Simulation.from_scene(SCENE, policy_path=POLICY)
                rollout(sim, 2)
                sim.set_command(forward=forward, yaw_rate=turn)
                moving = rollout(sim, 5)
                delta = moving[-1, :2] - moving[0, :2]
                yaw_delta = moving[-1, 2] - moving[0, 2]
                if forward:
                    self.assertGreater(delta[0], 0.15)
                    self.assertLess(abs(delta[1]), 0.10)
                else:
                    self.assertGreater(np.sign(turn) * yaw_delta, 1.0)
                    self.assertLess(np.linalg.norm(delta), 0.10)
                sim.set_command()
                stopped = rollout(sim, 5)
                speed, yaw_rate = self.assert_stopped(sim, stopped)
                print(
                    f"{label}: displacement={np.linalg.norm(delta):.3f}m, "
                    f"yaw={np.degrees(yaw_delta):.1f}deg, "
                    f"settled speed={speed:.4f}m/s, yaw rate={yaw_rate:.4f}rad/s"
                )

    def test_turn_then_forward_uses_robot_heading_and_stops(self):
        sim = Simulation.from_scene(SCENE, policy_path=POLICY)
        rollout(sim, 2)
        for turn in (TURN_SPEED, -TURN_SPEED):
            sim.set_command(yaw_rate=turn)
            turning = rollout(sim, 5)
            self.assertGreater(np.sign(turn) * (turning[-1, 2] - turning[0, 2]), 1.0)
            sim.set_command()
            self.assert_stopped(sim, rollout(sim, 5))
            sim.set_command(forward=FORWARD_SPEED)
            moving = rollout(sim, 5)
            heading = moving[0, 2]
            forward = np.array([np.cos(heading), np.sin(heading)])
            delta = moving[-1, :2] - moving[0, :2]
            self.assertGreater(np.dot(delta, forward), 0.15)
            self.assertLess(abs(np.dot(delta, [-forward[1], forward[0]])), 0.10)
            sim.set_command()
            self.assert_stopped(sim, rollout(sim, 5))

    def test_stands_for_30_seconds_and_resets_reproducibly(self):
        sim = Simulation.from_scene(SCENE, policy_path=POLICY)
        first = []
        for index in range(6000):
            sim.step()
            if index < 100:
                first.append(sim.data.qpos.copy())
        self.assertAlmostEqual(sim.data.time, 30)
        self.assertLess(sim.max_displacement, 0.10)
        print(
            f"Real policy: height >= {sim.min_height:.4f}m, "
            f"tilt <= {sim.max_tilt:.2f}deg, drift <= {sim.max_displacement:.4f}m"
        )
        sim.reset()
        for expected in first:
            sim.step()
            np.testing.assert_allclose(sim.data.qpos, expected, atol=1e-10, rtol=0)


if __name__ == "__main__":
    unittest.main()
