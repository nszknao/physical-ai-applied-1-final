"""MuJoCo state, BAM motor control, and paused/headless execution."""

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, SimpleQueue

import mujoco
import mujoco.viewer
import numpy as np
from bam.model import Model as BamModel
from bam.model import load_model
from bam.mujoco import MujocoController

from duckscout.locomotion import (
    CONTROL_DECIMATION,
    DEFAULT_POSE,
    FORWARD_SPEED,
    JOINT_NAMES,
    PHYSICS_TIMESTEP,
    TURN_SPEED,
    LocomotionPolicy,
)

# Flat-floor posture limits, independent of distance traveled.
MIN_UPRIGHT_HEIGHT = 0.08
MAX_UPRIGHT_TILT = 45.0
CAMERA_NAME = "head_camera"
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_PERIOD = 0.1  # Simulation seconds; independent of the viewer refresh rate.


def configure_bam(spec: mujoco.MjSpec) -> BamModel:
    """Configure the scene for XL330 M6 motors and return their BAM model."""
    # Mirror upstream's CPU rehearsal: voltage motor + M6 friction.
    bam = load_model(motor_name="xl330", model="m6")
    bam.actuator.kp = 200.0
    bam.actuator.vin = 7.4
    bam.actuator.max_current = None
    force_limit = 7.4 * bam.kt.value / bam.R.value
    if tuple(act.name for act in spec.actuators) != JOINT_NAMES:
        raise ValueError("Scene actuator order does not match velstand")
    for actuator in spec.actuators:
        if actuator.target != actuator.name:
            raise ValueError(f"Unexpected joint target: {actuator.name}")
        actuator.set_to_motor()
        actuator.forcelimited = True
        actuator.forcerange = (-force_limit, force_limit)
        actuator.ctrllimited = False
        actuator.gear = [1, 0, 0, 0, 0, 0]
    for joint in spec.joints:
        if joint.name in JOINT_NAMES:
            joint.damping = np.zeros((3, 1))
            joint.frictionloss = 0.0
            joint.solref_friction = (-5e4, -2e2)
            joint.solimp_friction = (0.99, 0.9999, 0.001, 0.5, 2.0)
    return bam


@dataclass(kw_only=True, eq=False)
class Simulation:
    """Own physics and control state; use from_scene() to assemble dependencies."""

    # Injected dependencies. BAM and policy are supplied together for locomotion.
    model: mujoco.MjModel
    data: mujoco.MjData
    key_id: int
    bam: BamModel | None = None
    policy: LocomotionPolicy | None = None

    # Runtime state, initialized here and restored by reset().
    motors: MujocoController | None = field(init=False, default=None)
    step_count: int = field(init=False, default=0)
    min_height: float = field(init=False, default=float("inf"))
    max_tilt: float = field(init=False, default=0.0)
    max_displacement: float = field(init=False, default=0.0)
    start_xy: np.ndarray = field(init=False, default_factory=lambda: np.zeros(2))

    def __post_init__(self) -> None:
        if (self.bam is None) != (self.policy is None):
            raise ValueError("BAM and locomotion policy must be supplied together")
        self.reset()

    @classmethod
    def from_scene(
        cls, path: Path, *, keyframe: str = "STAND", policy_path: Path | None = None
    ) -> "Simulation":
        if policy_path is not None and keyframe != "STAND":
            raise ValueError("Locomotion policy requires --keyframe STAND")
        bam = None
        policy = None
        spec = mujoco.MjSpec.from_file(str(path.resolve()))
        head_camera = spec.camera(CAMERA_NAME)
        if head_camera is not None and np.allclose(head_camera.quat, (0, 0, -1, 0)):
            # The upstream camera looks backward with a sideways image. In the
            # jaw frame, -Z is forward and +X is up: rotate camera axes -90° on Z.
            head_camera.quat = (np.sqrt(0.5), 0, 0, -np.sqrt(0.5))
        if policy_path is not None:
            bam = configure_bam(spec)
            spec.option.timestep = PHYSICS_TIMESTEP
        model = spec.compile()
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id < 0:
            raise ValueError(f"Unknown keyframe: {keyframe}")
        if not np.isfinite(model.opt.timestep) or model.opt.timestep <= 0:
            raise ValueError("The model must have a positive finite timestep")
        if policy_path is not None:
            policy = LocomotionPolicy.from_onnx(model, policy_path)
        simulation = cls(model=model, data=data, key_id=key_id, bam=bam, policy=policy)
        print(f"Loaded {path.name}: nq={model.nq}, nv={model.nv}, actuators={model.nu}")
        if policy is not None:
            print(
                "Locomotion: CPU policy 50 Hz | BAM XL330 M6 / physics 200 Hz | zero command"
            )

        return simulation

    def reset(self):
        if self.policy is not None:
            # BAM changes model friction as well as MjData; reset both.
            self.model.dof_damping[self.policy.qvel_indices] = 0
            self.model.dof_frictionloss[self.policy.qvel_indices] = 0
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        if self.policy is not None:
            self.bam.reset()
            # Recreate through the public API to clear all controller history.
            self.motors = MujocoController(
                self.bam,
                list(JOINT_NAMES),
                self.model,
                self.data,
                vin_drop_gain=0.1,
                vin_min=6.0,
            )
            root = self.policy.root_qpos
            self.data.qpos[root : root + 7] = (0, 0, 0.125, 1, 0, 0, 0)
            self.data.qpos[self.policy.qpos_indices] = DEFAULT_POSE
            self.data.ctrl[:] = 0
            self.policy.reset()
            self.motors.reset(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.min_height = float("inf")
        self.max_tilt = 0.0
        self.max_displacement = 0.0
        if self.policy is not None:
            root = self.policy.root_qpos
            self.start_xy = self.data.qpos[root : root + 2].copy()
            self.check_upright()

    def set_command(self, *, forward: float = 0.0, yaw_rate: float = 0.0) -> None:
        if self.policy is None:
            raise ValueError("Velocity commands require --policy")
        self.policy.set_command(forward=forward, yaw_rate=yaw_rate)

    def step(self):
        if self.policy is not None:
            if self.step_count % CONTROL_DECIMATION == 0:
                self.motors.q_target[:] = self.policy.targets(self.data)
            self.motors.update()
        previous_time = self.data.time
        mujoco.mj_step(self.model, self.data)
        if self.data.time <= previous_time or not all(
            np.isfinite(values).all()
            for values in (self.data.qpos, self.data.qvel, self.data.qacc)
        ):
            raise RuntimeError("Physics became unstable; stop and reset the simulation")
        for warning in (
            mujoco.mjtWarning.mjWARN_BADQPOS,
            mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC,
            mujoco.mjtWarning.mjWARN_BADCTRL,
        ):
            if self.data.warning[warning].number:
                raise RuntimeError(f"MuJoCo reported {warning.name}")
        self.step_count += 1
        if self.policy is not None:
            self.check_upright()

    def check_upright(self):
        root = self.policy.root_qpos
        position = self.data.qpos[root : root + 3]
        w, x, y, z = self.data.qpos[root + 3 : root + 7]
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
        drift = np.linalg.norm(position[:2] - self.start_xy)
        self.min_height = min(self.min_height, float(position[2]))
        self.max_tilt = max(self.max_tilt, float(tilt))
        self.max_displacement = max(self.max_displacement, float(drift))
        if position[2] < MIN_UPRIGHT_HEIGHT or tilt > MAX_UPRIGHT_TILT:
            raise RuntimeError(
                f"Posture check failed at {self.data.time:.3f}s: "
                f"height={position[2]:.3f}m, tilt={tilt:.1f}deg, drift={drift:.3f}m"
            )


def render_head_camera(simulation: Simulation, renderer: mujoco.Renderer) -> np.ndarray:
    """Read RGB at the current pose without advancing physics or the policy."""
    # mj_step leaves position-derived fields at the previous integration state.
    # Refresh only poses, avoiding another dynamics/constraint solver evaluation.
    mujoco.mj_kinematics(simulation.model, simulation.data)
    mujoco.mj_camlight(simulation.model, simulation.data)
    renderer.update_scene(simulation.data, camera=CAMERA_NAME)
    return renderer.render()


def save_snapshot(path: Path, rgb: np.ndarray) -> None:
    """Save lossless RGB as binary PPM without an image-library dependency."""
    if path.suffix.lower() != ".ppm":
        raise ValueError("Snapshot path must end in .ppm")
    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8:
        raise ValueError("Snapshot must be an RGB uint8 image")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes())
    print(f"Saved head camera: {path} ({width}x{height} RGB)")


def run_headless(simulation: Simulation, steps: int, *, snapshot: Path | None = None):
    for _ in range(steps):
        simulation.step()
    print(f"Physics OK: {steps} steps, simulation time={simulation.data.time:.3f}s")
    if simulation.policy is not None:
        print(
            f"Posture OK: min height={simulation.min_height:.3f}m, "
            f"max tilt={simulation.max_tilt:.2f}deg, "
            f"max displacement={simulation.max_displacement:.4f}m"
        )
    if snapshot is not None:
        with mujoco.Renderer(
            simulation.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH
        ) as renderer:
            save_snapshot(snapshot, render_head_camera(simulation, renderer))


# GLFW arrow key codes supplied by MuJoCo's public key_callback.
MOTION_KEYS = {
    265: ("Forward", FORWARD_SPEED, 0.0),  # Up
    263: ("Turn left", 0.0, TURN_SPEED),  # Left
    262: ("Turn right", 0.0, -TURN_SPEED),  # Right
    264: ("Stop (balance remains active)", 0.0, 0.0),  # Down
}


def handle_key(simulation: Simulation, key: int, paused: bool) -> bool:
    """Apply a queued key on the physics thread and return the pause state."""
    if key == 32:
        paused = not paused
        print("Paused" if paused else "Running")
    elif key == ord("R"):
        simulation.reset()
        paused = True
        print("Reset; zero command; paused")
    elif key in MOTION_KEYS and simulation.policy is not None:
        label, forward, yaw_rate = MOTION_KEYS[key]
        simulation.set_command(forward=forward, yaw_rate=yaw_rate)
        print(
            f"{label}: forward={forward:.2f}m/s, yaw={yaw_rate:.2f}rad/s"
            + ("; paused (Space to run)" if paused else "")
        )
    return paused


def run_viewer(
    simulation: Simulation,
    *,
    camera: str = "follow",
    snapshot: Path = Path("runs/head_camera.ppm"),
):
    # GUI callbacks only enqueue commands; the main loop owns physics state.
    commands = SimpleQueue()
    paused = True
    model, data = simulation.model, simulation.data
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if camera == "head" and camera_id < 0:
        raise ValueError(f"Scene has no {CAMERA_NAME} camera")
    print("Paused. Space: run/pause | R: reset and pause | Esc/window close: exit")
    if simulation.policy is None:
        print("No walking policy is active. The robot may fall when physics runs.")
    else:
        print("Up: forward | Left/Right: turn in place | Down: stop and balance")
        print("Commands persist until the next arrow key; Space only pauses time.")
    if camera_id >= 0:
        print("Rendering > Camera: Tracking/head_camera | Enter: save head-camera PPM")
    with (
        (
            mujoco.Renderer(model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH)
            if camera_id >= 0
            else nullcontext()
        ) as renderer,
        mujoco.viewer.launch_passive(model, data, key_callback=commands.put) as viewer,
    ):
        with viewer.lock():
            viewer.cam.lookat[:] = (0, 0, 0.15)
            viewer.cam.distance = 0.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            if simulation.policy is not None:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = simulation.policy.trunk_id
                # Native Tracking selection uses the selected body.
                viewer.perturb.select = simulation.policy.trunk_id
            if camera == "head":
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id
        last_sync = 0.0
        next_capture = 0.0
        rgb = None
        while viewer.is_running():
            started = time.monotonic()
            save_requested = False
            with viewer.lock():
                while True:
                    try:
                        key = commands.get_nowait()
                    except Empty:
                        break
                    if key == 256:  # Escape
                        return
                    if key == 257 and renderer is not None:  # Enter
                        save_requested = True
                    if key == ord("R"):
                        next_capture = 0.0
                    paused = handle_key(simulation, key, paused)
                if not paused:
                    simulation.step()
                if renderer is not None and (
                    save_requested or data.time + 1e-9 >= next_capture
                ):
                    rgb = render_head_camera(simulation, renderer)
                    next_capture = data.time + CAMERA_PERIOD
            if save_requested:
                save_snapshot(snapshot, rgb)
            if started - last_sync >= 1 / 60:
                viewer.sync()
                last_sync = started
            period = 1 / 60 if paused else model.opt.timestep
            time.sleep(max(0, period - (time.monotonic() - started)))
