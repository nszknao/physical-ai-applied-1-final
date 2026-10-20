"""Shared robot assets and fake ONNX setup; contains no test cases."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
