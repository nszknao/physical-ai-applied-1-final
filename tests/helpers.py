"""Shared robot assets and fake ONNX setup; contains no test cases."""

from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import mujoco
import numpy as np

from duckscout.locomotion import JOINT_NAMES, PHYSICS_TIMESTEP, LocomotionPolicy
from duckscout.simulation import Simulation, configure_bam

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


def make_simulation(session):
    """Assemble the real physics model with a caller-supplied inference session."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    bam = configure_bam(spec)
    spec.option.timestep = PHYSICS_TIMESTEP
    model = spec.compile()
    policy = LocomotionPolicy(model=model, session=session)
    return Simulation(
        model=model,
        data=mujoco.MjData(model),
        key_id=model.key("STAND").id,
        bam=bam,
        policy=policy,
    )


class ProjectedCubeRenderer:
    """Test double: projects target geometry into RGB; does not emulate shading/occlusion."""

    def __init__(self, model, visible=True):
        self.model = model
        self.visible = visible

    def update_scene(self, data, camera):
        self.data = data
        self.camera = self.model.camera(camera).id

    def render(self):
        image = np.zeros((240, 320, 3), np.uint8)
        if not self.visible:
            return image
        m, d, c = self.model, self.data, self.camera
        target = m.geom("target").id
        corners = np.array(list(product((-1, 1), repeat=3))) * m.geom_size[target]
        world = corners @ d.geom_xmat[target].reshape(3, 3).T + d.geom_xpos[target]
        local = (world - d.cam_xpos[c]) @ d.cam_xmat[c].reshape(3, 3)
        if np.any(local[:, 2] >= -0.005):
            return image
        f = 120 / np.tan(np.radians(m.cam_fovy[c]) / 2)
        pixels = np.stack(
            (
                160 + f * local[:, 0] / -local[:, 2],
                120 - f * local[:, 1] / -local[:, 2],
            ),
            axis=1,
        )
        hull = cv2.convexHull(np.round(pixels).astype(np.int32))
        cv2.fillConvexPoly(image, hull, (255, 0, 0))
        return image


def projected_penalty_observation(simulation, ball_name="ball"):
    """Perception test double using projection, NOT VLM/occlusion verification.

    Test-only ground truth supplies observations while the real walking policy and
    ball contacts run. Production steering uses camera-based VLM answers and color tracking.
    """
    model, data = simulation.model, simulation.data
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    camera = data.camera("head_camera")
    focal = 120 / np.tan(np.deg2rad(model.camera("head_camera").fovy[0] / 2))

    def project(point):
        local = camera.xmat.reshape(3, 3).T @ (point - camera.xpos)
        depth = -local[2]
        return np.array(
            [160 + focal * local[0] / depth, 120 - focal * local[1] / depth]
        ), depth

    def position(x):
        return "left" if x < 320 / 3 else "right" if x > 640 / 3 else "center"

    qpos = model.joint(f"{ball_name}_freejoint").qposadr[0]
    pixel, depth = project(data.qpos[qpos : qpos + 3])
    goal, _ = project(np.array([1.0, 0.0, 0.12]))
    radius = focal * model.geom(ball_name).size[0] / depth
    visible = depth > 0 and pixel[1] - radius < 240
    return {
        "ball_position": position(pixel[0]) if visible else "not_visible",
        "goal_position": position(goal[0]),
        "ball_inside_image": bool(
            depth > 0
            and radius <= pixel[0] <= 320 - radius
            and radius <= pixel[1] <= 240 - radius
        ),
        "ball_at_bottom": bool(
            depth > 0
            and 240 - radius < pixel[1] < 240 + radius
            and 0 <= pixel[0] <= 320
        ),
    }
