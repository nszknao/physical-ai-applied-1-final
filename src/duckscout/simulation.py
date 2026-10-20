"""Load the upstream scene and inspect its untrained physical behavior."""

from pathlib import Path
from queue import Empty, SimpleQueue
import time

import mujoco
import mujoco.viewer
import numpy as np


def load_scene(path: Path, keyframe: str):
    model = mujoco.MjModel.from_xml_path(str(path.resolve()))
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key_id < 0:
        raise ValueError(f"Unknown keyframe: {keyframe}")
    if not np.isfinite(model.opt.timestep) or model.opt.timestep <= 0:
        raise ValueError("The model must have a positive finite timestep")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    print(f"Loaded {path.name}: nq={model.nq}, nv={model.nv}, actuators={model.nu}")
    return model, data, key_id


def step(model, data):
    previous_time = data.time
    mujoco.mj_step(model, data)
    if data.time <= previous_time or not all(
        np.isfinite(values).all() for values in (data.qpos, data.qvel, data.qacc)
    ):
        raise RuntimeError("Physics became unstable; stop and reset the simulation")
    for warning in (
        mujoco.mjtWarning.mjWARN_BADQPOS,
        mujoco.mjtWarning.mjWARN_BADQVEL,
        mujoco.mjtWarning.mjWARN_BADQACC,
        mujoco.mjtWarning.mjWARN_BADCTRL,
    ):
        if data.warning[warning].number:
            raise RuntimeError(f"MuJoCo reported {warning.name}")


def run_headless(model, data, steps: int):
    for _ in range(steps):
        step(model, data)
    print(f"Physics OK: {steps} steps, simulation time={data.time:.3f}s")


def run_viewer(model, data, key_id: int):
    # GUI callbacks only enqueue commands; the main loop owns physics state.
    commands = SimpleQueue()
    paused = True
    print("Paused. Space: run/pause | R: reset and pause | Esc/window close: exit")
    print("No walking policy is active. The robot may fall when physics runs.")
    with mujoco.viewer.launch_passive(model, data, key_callback=commands.put) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = (0, 0, 0.15)
            viewer.cam.distance = 0.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
        last_sync = 0.0
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                while True:
                    try:
                        key = commands.get_nowait()
                    except Empty:
                        break
                    if key == 32:
                        paused = not paused
                        print("Paused" if paused else "Running")
                    elif key == ord("R"):
                        mujoco.mj_resetDataKeyframe(model, data, key_id)
                        mujoco.mj_forward(model, data)
                        paused = True
                        print("Reset; paused")
                if not paused:
                    step(model, data)
            if started - last_sync >= 1 / 60:
                viewer.sync()
                last_sync = started
            period = 1 / 60 if paused else model.opt.timestep
            time.sleep(max(0, period - (time.monotonic() - started)))
