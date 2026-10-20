"""The official velstand observation layout and CPU inference.

Reference: pollen-robotics/microduck_rl scripts/infer_policy.py, revision
b02fdea4902c2b93b7d34a83a6f05da4ececb7d3 (Apache-2.0, Pollen Robotics).
"""

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
DEFAULT_POSE = np.array(
    [
        0,
        -0.0873,
        -0.4579,
        -0.0049,
        0.4530,
        0.3491,
        0.3491,
        0,
        0,
        0,
        0.0873,
        0.4579,
        0.0049,
        -0.4530,
    ],
    dtype=np.float32,
)
JOINT_COUNT = len(JOINT_NAMES)
COMMAND_SIZE = 3 + 4 + 6  # twist, head pose, body pose
OBSERVATION_SIZE = 3 + 3 + 3 * JOINT_COUNT + COMMAND_SIZE
CONTROL_DECIMATION = 4
PHYSICS_TIMESTEP = 0.005
FORWARD_SPEED = 0.30  # m/s, also the maximum accepted forward command
TURN_SPEED = 0.80  # rad/s, also the maximum accepted absolute yaw command


@dataclass(kw_only=True, eq=False)
class LocomotionPolicy:
    """Validate dependencies and translate robot state into locomotion targets."""

    # Injected dependencies. from_onnx() loads a CPU session for normal use.
    model: mujoco.MjModel
    session: ort.InferenceSession

    # Addresses and names derived from the validated model and session.
    qpos_indices: np.ndarray = field(init=False)
    qvel_indices: np.ndarray = field(init=False)
    trunk_id: int = field(init=False)
    root_qpos: int = field(init=False)
    gyro_address: int = field(init=False)
    input_name: str = field(init=False)
    output_name: str = field(init=False)

    # Per-instance history, cleared on reset.
    last_action: np.ndarray = field(
        init=False, default_factory=lambda: np.zeros(JOINT_COUNT, dtype=np.float32)
    )

    command: np.ndarray = field(
        init=False, default_factory=lambda: np.zeros(COMMAND_SIZE, dtype=np.float32)
    )

    def __post_init__(self) -> None:
        model = self.model
        # Validate the transmission, not just the number of actuators.
        if model.nu != JOINT_COUNT:
            raise ValueError(
                f"velstand requires exactly {JOINT_COUNT} MicroDuck actuators"
            )
        joints = []
        for index, name in enumerate(JOINT_NAMES):
            joint = model.joint(name)
            actuator = model.actuator(index)
            if (
                joint.type[0] != mujoco.mjtJoint.mjJNT_HINGE
                or actuator.name != name
                or actuator.trntype[0] != mujoco.mjtTrn.mjTRN_JOINT
                or actuator.trnid[0] != joint.id
            ):
                raise ValueError(
                    f"Incompatible actuator at index {index}: expected {name}"
                )
            joints.append(joint.id)
        self.qpos_indices = model.jnt_qposadr[joints].copy()
        self.qvel_indices = model.jnt_dofadr[joints].copy()
        self.trunk_id = model.body("trunk_base").id
        root = model.joint("trunk_base_freejoint")
        if root.type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("trunk_base_freejoint must be a free joint")
        self.root_qpos = int(root.qposadr[0])
        sensor = model.sensor("imu_ang_vel")
        if sensor.dim[0] != 3 or sensor.type[0] != mujoco.mjtSensor.mjSENS_GYRO:
            raise ValueError("imu_ang_vel must be a 3D gyroscope")
        self.gyro_address = int(sensor.adr[0])

        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        for nodes, width, label in (
            (inputs, OBSERVATION_SIZE, "input"),
            (outputs, JOINT_COUNT, "output"),
        ):
            if (
                len(nodes) != 1
                or nodes[0].type != "tensor(float)"
                or len(nodes[0].shape) != 2
                or nodes[0].shape[1] != width
                or (isinstance(nodes[0].shape[0], int) and nodes[0].shape[0] != 1)
            ):
                raise ValueError(f"velstand {label} must be float32 [1, {width}]")
        metadata = self.session.get_modelmeta().custom_metadata_map
        expected_metadata = {
            "joint_names": ",".join(JOINT_NAMES),
            "observation_names": (
                "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,"
                "command,head_command,body_command"
            ),
            "command_names": "twist,head_pose,body_pose",
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(f"Incompatible policy metadata: {name}")
        if float(metadata.get("action_scale", "nan")) != 1.0:
            raise ValueError("Locomotion policy requires action_scale = 1.0")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    @classmethod
    def from_onnx(cls, model: mujoco.MjModel, path: Path) -> "LocomotionPolicy":
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        try:
            session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise RuntimeError(f"Cannot load ONNX policy {path}: {error}") from error
        return cls(model=model, session=session)

    def set_command(self, *, forward: float = 0.0, yaw_rate: float = 0.0) -> None:
        """Set body-frame forward speed [m/s] and yaw rate [rad/s]."""
        if not np.isfinite([forward, yaw_rate]).all():
            raise ValueError("Velocity commands must be finite")
        if not 0 <= forward <= FORWARD_SPEED or abs(yaw_rate) > TURN_SPEED:
            raise ValueError(
                f"Commands require forward in [0, {FORWARD_SPEED}] m/s "
                f"and yaw_rate in [-{TURN_SPEED}, {TURN_SPEED}] rad/s"
            )
        self.command.fill(0)
        self.command[0] = forward
        self.command[2] = yaw_rate

    def reset(self) -> None:
        self.last_action.fill(0)
        self.command.fill(0)

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        # xmat maps body to world; its last row negated is gravity in body axes.
        gravity = -data.xmat[self.trunk_id].reshape(3, 3)[2]
        observation = np.concatenate(
            (
                data.sensordata[self.gyro_address : self.gyro_address + 3],
                gravity,
                data.qpos[self.qpos_indices] - DEFAULT_POSE,
                data.qvel[self.qvel_indices],
                self.last_action,
                self.command,  # twist(3), head(4), body(6); zero means stand.
            )
        ).astype(np.float32)
        if not np.isfinite(observation).all():
            raise RuntimeError("Policy observation contains non-finite values")
        return observation[None, :]

    def targets(self, data: mujoco.MjData) -> np.ndarray:
        observation = self.observation(data)
        try:
            action = self.session.run(
                [self.output_name], {self.input_name: observation}
            )[0]
        except Exception as error:
            raise RuntimeError(f"ONNX inference failed: {error}") from error
        if action.shape != (1, JOINT_COUNT) or not np.isfinite(action).all():
            raise RuntimeError(
                f"Policy action must be finite with shape [1, {JOINT_COUNT}]"
            )
        # The next observation contains the raw output, not the position target.
        self.last_action[:] = action[0]
        return DEFAULT_POSE + self.last_action
