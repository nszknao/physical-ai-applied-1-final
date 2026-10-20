"""Tests for duckscout.simulation: physics, control, viewer and camera output."""

import tempfile
import unittest
from contextlib import nullcontext
from itertools import count
from pathlib import Path
from unittest.mock import Mock, patch

import mujoco
import numpy as np
from helpers import POLICY, SCENE, fake_session, make_simulation

from duckscout.locomotion import (
    DEFAULT_POSE,
    FORWARD_SPEED,
    TURN_SPEED,
)
from duckscout.simulation import (
    CAMERA_HEIGHT,
    CAMERA_NAME,
    CAMERA_WIDTH,
    Simulation,
    handle_key,
    render_head_camera,
    run_headless,
    run_viewer,
    save_snapshot,
    show_camera_preview,
)

SEARCH_SCENE = Path(__file__).resolve().parents[1] / "assets/scenes/search.xml"


@unittest.skipUnless(
    SCENE.is_file(), "Run scripts/fetch_assets.py to install the scene"
)
class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.session = fake_session()
        self.sim = make_simulation(self.session)

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


@unittest.skipUnless(
    SCENE.is_file() and POLICY.is_file(), "Fetch robot and policy assets first"
)
class CameraGeometryTests(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation.from_scene(SEARCH_SCENE, policy_path=POLICY)
        self.renderer = Mock()

    def project(self, point):
        model, data = self.sim.model, self.sim.data
        render_head_camera(self.sim, self.renderer)
        camera = model.camera(CAMERA_NAME).id
        rotation = data.cam_xmat[camera].reshape(3, 3)
        x, y, z = rotation.T @ (point - data.cam_xpos[camera])
        self.assertLess(z, 0, "Point must be in front of the camera")
        focal = CAMERA_HEIGHT / (2 * np.tan(np.radians(model.cam_fovy[camera]) / 2))
        return np.array(
            [CAMERA_WIDTH / 2 + focal * x / -z, CAMERA_HEIGHT / 2 - focal * y / -z]
        )

    def test_fixed_target_preserves_robot_dofs_and_is_visible_after_near_clip(self):
        model, data = self.sim.model, self.sim.data
        self.assertEqual((model.nq, model.nv, model.nu), (21, 20, 14))
        target = model.geom("target")
        self.assertEqual(target.bodyid[0], 0)
        np.testing.assert_allclose(target.size, [0.04] * 3)
        np.testing.assert_allclose(target.pos, [0.8, 0, 0.04])
        for steps in (0, 400):
            for _ in range(steps):
                self.sim.step()
            point = data.geom("target").xpos.copy()
            pixel = self.project(point)
            self.assertTrue((pixel > 0).all())
            self.assertTrue((pixel < [CAMERA_WIDTH, CAMERA_HEIGHT]).all())
            camera = model.camera(CAMERA_NAME).id
            rotation = data.cam_xmat[camera].reshape(3, 3)
            self.assertGreater((-rotation[:, 2])[0], 0.9)
            self.assertGreater(rotation[2, 1], 0.9)
            origin = data.cam_xpos[camera].copy()
            direction = point - origin
            direction /= np.linalg.norm(direction)
            # The lens lies just ahead of the camera origin, inside the near clip.
            near = model.vis.map.znear * model.stat.extent
            origin += direction * near / np.dot(direction, -rotation[:, 2])
            hit = np.full(1, -1, dtype=np.int32)
            distance = mujoco.mj_ray(
                model,
                data,
                origin,
                direction,
                np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8),
                True,
                -1,
                hit,
            )
            self.assertGreater(distance, 0)
            self.assertEqual(hit[0], target.id, "Target is occluded in the camera")

    def test_projection_has_correct_axes_and_grows_when_approaching(self):
        target = self.sim.data.geom("target").xpos.copy()
        center = self.project(target)
        left = self.project(target + [0, 0.02, 0])
        top = self.project(target + [0, 0, 0.02])
        self.assertLess(left[0], center[0])
        self.assertLess(top[1], center[1])
        far_width = np.linalg.norm(left - center)
        self.sim.data.qpos[self.sim.policy.root_qpos] += 0.2
        near_width = np.linalg.norm(
            self.project(target + [0, 0.02, 0]) - self.project(target)
        )
        self.assertGreater(near_width, far_width * 1.2)
        # Turning left moves the target right, eventually out of the image.
        root = self.sim.policy.root_qpos
        yaw = np.radians(45)
        self.sim.data.qpos[root + 3 : root + 7] = [
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        ]
        self.assertGreater(self.project(target)[0], CAMERA_WIDTH)
        self.sim.reset()
        np.testing.assert_allclose(self.project(target), center)

    def test_viewer_initializes_both_views_with_real_public_handle(self):
        for camera, expected_type in (
            ("head", mujoco.mjtCamera.mjCAMERA_FIXED),
            ("follow", mujoco.mjtCamera.mjCAMERA_TRACKING),
        ):
            with self.subTest(camera=camera):
                backend = Mock(
                    exitrequest=0, viewport=mujoco.MjrRect(100, 20, 800, 600)
                )
                backend.lock.return_value = nullcontext()
                backend.sync.side_effect = lambda _: setattr(backend, "exitrequest", 1)
                # Real Handle properties catch unsupported attribute names. Only
                # the window backend and GPU rendering are replaced for this test.
                viewer = mujoco.viewer.Handle(
                    backend,
                    mujoco.MjvCamera(),
                    mujoco.MjvOption(),
                    mujoco.MjvPerturb(),
                    None,
                )
                with (
                    patch(
                        "duckscout.simulation.mujoco.viewer.launch_passive",
                        return_value=viewer,
                    ),
                    patch("duckscout.simulation.mujoco.Renderer") as renderer,
                    patch("duckscout.simulation.time.sleep"),
                ):
                    renderer.return_value.__enter__.return_value.render.return_value = (
                        np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                    )
                    run_viewer(self.sim, camera=camera)
                self.assertEqual(viewer.cam.type, expected_type)
                self.assertEqual(viewer.cam.trackbodyid, self.sim.policy.trunk_id)
                self.assertEqual(viewer.perturb.select, self.sim.policy.trunk_id)
                if camera == "head":
                    self.assertEqual(
                        viewer.cam.fixedcamid, self.sim.model.camera(CAMERA_NAME).id
                    )
                backend.set_images.assert_called_once()
                backend.sync.assert_called_once()
                backend.exit.assert_called_once()
                renderer.return_value.__exit__.assert_called_once()

    def test_viewer_replaces_overlay_when_target_is_lost_and_reacquired(self):
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(3)]
        frames[0][80:100, 40:60] = (255, 0, 0)
        frames[2][80:100, 200:220] = (255, 0, 0)
        backend = Mock(exitrequest=0, viewport=mujoco.MjrRect(0, 0, 976, 600))
        backend.lock.return_value = nullcontext()
        viewer = mujoco.viewer.Handle(
            backend, mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvPerturb(), None
        )
        callbacks = []

        def launch(*args, key_callback):
            callbacks.append(key_callback)
            return viewer

        def sync(_):
            if backend.sync.call_count < 3:
                callbacks[0](257)  # Capture a fresh image even while paused.
            else:
                backend.exitrequest = 1

        backend.sync.side_effect = sync
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "duckscout.simulation.mujoco.viewer.launch_passive",
                    side_effect=launch,
                ),
                patch("duckscout.simulation.mujoco.Renderer") as renderer,
                patch("duckscout.simulation.time.sleep"),
                patch(
                    "duckscout.simulation.time.monotonic", side_effect=count(1, 0.02)
                ),
            ):
                renderer.return_value.__enter__.return_value.render.side_effect = frames
                run_viewer(self.sim, snapshot=Path(directory) / "head.ppm")
        self.assertEqual(backend.set_images.call_count, 3)
        for index, expected_x in enumerate((40, None, 200)):
            # The real Handle flips images for OpenGL; restore top-left origin.
            preview = np.flip(
                backend.set_images.call_args_list[index].args[0][0][1], axis=0
            )
            green = np.all(preview == (0, 255, 0), axis=2)
            ys, xs = np.nonzero(green)
            if expected_x is None:
                self.assertEqual(len(xs), 0)
            else:
                self.assertEqual(int(xs.min()), expected_x)
                self.assertEqual(int(ys.min()), 80)
        self.assertEqual(self.sim.data.time, 0)

    def test_render_delays_do_not_slow_fixed_step_physics(self):
        clock = [0.0]
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        backend = Mock(exitrequest=0, viewport=mujoco.MjrRect(0, 0, 976, 600))
        backend.lock.return_value = nullcontext()
        viewer = mujoco.viewer.Handle(
            backend, mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvPerturb(), None
        )

        def advance(seconds):
            clock[0] += seconds
            if clock[0] > 3:
                self.fail("Viewer failed to keep up with the simulated clock")

        def render():
            advance(0.03)  # GPU readback takes 30 ms, longer than a 5 ms step.
            return image

        def launch(*args, key_callback):
            key_callback(32)  # Start physics immediately.
            return viewer

        def sync(_):
            advance(0.001)
            if self.sim.data.time >= 1.0:
                backend.exitrequest = 1

        backend.sync.side_effect = sync
        backend.set_images.side_effect = lambda _: advance(0.03)
        with (
            patch(
                "duckscout.simulation.mujoco.viewer.launch_passive", side_effect=launch
            ),
            patch("duckscout.simulation.mujoco.Renderer") as renderer,
            patch("duckscout.simulation.time.monotonic", side_effect=lambda: clock[0]),
            patch("duckscout.simulation.time.sleep", side_effect=advance),
        ):
            renderer.return_value.__enter__.return_value.render.side_effect = render
            run_viewer(self.sim)
        self.assertGreaterEqual(self.sim.data.time, 1.0)
        self.assertLess(clock[0], 1.15)
        self.assertLess(abs(clock[0] - self.sim.data.time), 0.1)
        self.assertAlmostEqual(self.sim.data.time, self.sim.step_count * 0.005)
        self.assertLessEqual(backend.set_images.call_count, 12)
        self.assertGreater(backend.sync.call_count, backend.set_images.call_count)

    def test_paused_preview_is_uploaded_only_once_and_again_on_resize(self):
        clock = [0.0]
        backend = Mock(exitrequest=0, viewport=mujoco.MjrRect(0, 0, 800, 600))
        backend.lock.return_value = nullcontext()
        viewer = mujoco.viewer.Handle(
            backend, mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvPerturb(), None
        )

        def sync(_):
            if backend.sync.call_count == 3:
                backend.viewport = mujoco.MjrRect(0, 0, 1000, 600)
            if backend.sync.call_count == 5:
                backend.exitrequest = 1

        backend.sync.side_effect = sync
        with (
            patch(
                "duckscout.simulation.mujoco.viewer.launch_passive", return_value=viewer
            ),
            patch("duckscout.simulation.mujoco.Renderer") as renderer,
            patch("duckscout.simulation.time.monotonic", side_effect=lambda: clock[0]),
            patch(
                "duckscout.simulation.time.sleep",
                side_effect=lambda seconds: clock.__setitem__(
                    0, clock[0] + max(seconds, 0.001)
                ),
            ),
        ):
            renderer.return_value.__enter__.return_value.render.return_value = np.zeros(
                (240, 320, 3), dtype=np.uint8
            )
            run_viewer(self.sim)
        renderer.return_value.__enter__.return_value.render.assert_called_once()
        self.assertEqual(backend.set_images.call_count, 2)
        self.assertEqual(self.sim.data.time, 0)

    def test_headless_saves_raw_and_annotated_images_separately(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[80:100, 100:120] = (255, 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.ppm"
            with patch("duckscout.simulation.mujoco.Renderer") as renderer:
                renderer.return_value.__enter__.return_value.render.return_value = image
                run_headless(self.sim, 2, snapshot=path)
            raw = path.read_bytes().split(b"\n", 3)[3]
            annotated = (
                path.with_stem("camera_detection").read_bytes().split(b"\n", 3)[3]
            )
            self.assertEqual(raw, image.tobytes())
            self.assertNotEqual(annotated, raw)
            pixels = np.frombuffer(annotated, dtype=np.uint8).reshape(image.shape)
            np.testing.assert_array_equal(pixels[80, 100], (0, 255, 0))

    def test_sampling_camera_does_not_change_control_trajectory(self):
        control = Simulation.from_scene(SEARCH_SCENE, policy_path=POLICY)
        self.sim.set_command(forward=0.3)
        control.set_command(forward=0.3)
        for step in range(400):
            self.sim.step()
            control.step()
            if step % 20 == 0:
                render_head_camera(self.sim, self.renderer)
        np.testing.assert_array_equal(self.sim.data.qpos, control.data.qpos)
        np.testing.assert_array_equal(self.sim.data.qvel, control.data.qvel)
        np.testing.assert_array_equal(
            self.sim.policy.last_action, control.policy.last_action
        )
        self.assertEqual(self.sim.data.time, control.data.time)


class CameraPreviewTests(unittest.TestCase):
    def test_overlay_fits_viewport_and_real_handle_flips_rgb_once(self):
        for width, height in ((800, 600), (200, 80), (2400, 1800)):
            with self.subTest(width=width, height=height):
                backend = Mock(viewport=mujoco.MjrRect(100, 40, width, height))
                viewer = mujoco.viewer.Handle(
                    backend,
                    mujoco.MjvCamera(),
                    mujoco.MjvOption(),
                    mujoco.MjvPerturb(),
                    None,
                )
                image = np.zeros((240, 320, 3), dtype=np.uint8)
                image[:120] = (255, 0, 0)
                image[120:] = (0, 0, 255)
                show_camera_preview(viewer, image)
                rect, pixels = backend.set_images.call_args.args[0][0]
                self.assertEqual(pixels.shape, (rect.height, rect.width, 3))
                self.assertGreaterEqual(rect.left, 100)
                self.assertGreaterEqual(rect.bottom, 40)
                self.assertLessEqual(rect.left + rect.width, 100 + width)
                self.assertLessEqual(rect.bottom + rect.height, 40 + height)
                np.testing.assert_array_equal(pixels[0, 0], (0, 0, 255))
                np.testing.assert_array_equal(pixels[-1, 0], (255, 0, 0))
                self.assertTrue(pixels.flags.c_contiguous)

    def test_minimized_viewport_clears_the_overlay(self):
        backend = Mock(viewport=mujoco.MjrRect(0, 0, 0, 0))
        viewer = mujoco.viewer.Handle(
            backend, mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvPerturb(), None
        )
        show_camera_preview(viewer, np.zeros((240, 320, 3), dtype=np.uint8))
        backend.clear_images.assert_called_once()
        backend.set_images.assert_not_called()


class SnapshotTests(unittest.TestCase):
    def test_ppm_preserves_rgb_order_rows_and_creates_parent_directory(self):
        rgb = np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [1, 2, 3]]], dtype=np.uint8
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames/head.ppm"
            save_snapshot(path, rgb)
            self.assertEqual(path.read_bytes(), b"P6\n2 2\n255\n" + rgb.tobytes())


if __name__ == "__main__":
    unittest.main()
