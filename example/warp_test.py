# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

import judo
import tyro
from dataclasses import dataclass
from typing import Optional
import mujoco_warp as mjw
import mujoco
import warp as wp
import time
import numpy as np

from typing import Optional
wp.init()


NCONMAX = 24
NJMAX = 64
@dataclass
class WarpTestConfig:
    model_path: str = "/home/slecleach/workspace/slecleach/judo/judo/models/xml/cylinder_push.xml"
    sim_model_path: str = "/home/slecleach/workspace/slecleach/judo/judo/models/xml/cylinder_push.xml"
    num_worlds: int = 8192
    num_steps: int = 10000

state_integration = [
    "time",
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
]





def main(config: WarpTestConfig) -> None:


    spec = mujoco.MjSpec.from_file(WarpTestConfig.model_path)
    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(mj_model)
        data = mjw.put_data(mj_model, mj_data, nworld=config.num_worlds, nconmax=NCONMAX, njmax=NJMAX)

    # capture the whole function as a CUDA graph
    jit_beg = time.perf_counter()
    with wp.ScopedCapture() as capture:
        mjw.step(model, data)
    jit_end = time.perf_counter()
    jit_duration = jit_end - jit_beg

    graph = capture.graph
        
    time_vec = np.zeros(config.num_steps)
    for i in range(config.num_steps):
        run_beg = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize()
        run_end = time.perf_counter()
        time_vec[i] = (run_end - run_beg) 

    # Create time, qpos, qvel arrays
    time_vals = 1.0 * np.ones((config.num_worlds, 1))
    qpos_vals = 2.0 * np.ones((config.num_worlds, mj_model.nq))
    qvel_vals = 3.0 * np.ones((config.num_worlds, mj_model.nv)) 
    ctrl_vals = 4.0 * np.ones((config.num_worlds, mj_model.nu))
    
    # Concatenate [time, qpos, qvel] into state array
    # Shape: (num_worlds, 1 + nq + nv)
    state_array_np = np.concatenate([
        time_vals,           # time: (num_worlds, 1)
        qpos_vals,           # qpos: (num_worlds, nq)
        qvel_vals            # qvel: (num_worlds, nv)
    ], axis=1)
    state_array = wp.array(state_array_np, dtype=wp.float32)
    ctrl_array = wp.array(ctrl_vals, dtype=wp.float32)

    # Create signature for TIME | QPOS | QVEL
    # https://github.com/google-deepmind/mujoco_warp/blob/10dc8f3890b2d9db87d82bf25efda06ac04798c3/mujoco_warp/_src/types.py#L538
    sig = mjw.State.TIME | mjw.State.QPOS | mjw.State.QVEL
    sig = mjw.State.FULLPHYSICS
    
    # Call set_state with time, qpos, qvel only
    mjw.set_state(model, data, state_array, sig)
    mjw.set_state(model, data, ctrl_array, mjw.State.CTRL)
    
    print("state_array: ", state_array)
    print("sig: ", sig)
    print("time_vals: ", time_vals)
    print("qpos_vals: ", qpos_vals)
    print("qvel_vals: ", qvel_vals)
    print("ctrl_vals: ", ctrl_vals)
    print("data.time: ", data.time)
    print("data.qpos: ", data.qpos)
    print("data.qvel: ", data.qvel)
    print("data.ctrl: ", data.ctrl)

    per_thread_physics_time = mj_model.opt.timestep * config.num_steps
    per_thread_rtf = per_thread_physics_time / np.sum(time_vec)
    rtf = per_thread_physics_time * config.num_worlds / np.sum(time_vec)

    print(f"JIT duration: {jit_duration:.3f} s")
    print(f"Average run duration: {np.mean(time_vec * 1e6):.3f} us")
    print(f"Total time: {np.sum(time_vec):.3f} s")
    print(f"Real time factor: {rtf:.3f}")
    print(f"Per thread real time factor: {per_thread_rtf:.3f}")

if __name__ == "__main__":
    main(tyro.cli(WarpTestConfig))