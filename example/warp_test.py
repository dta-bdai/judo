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

# # Import State, Model, Data, nested_kernel from mujoco_warp
# State = mjw.State
# Model = mjw.Model
# Data = mjw.Data
# nested_kernel = mjw.nested_kernel
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



# def set_state(m: mjw.Model, d: mjw.Data, state: wp.array2d(dtype=float), sig: int, active: Optional[wp.array] = None):
#   """Copy concatenated state components specified by sig from state into Data.

#   The bits of the integer sig correspond to element fields of State.

#   Args:
#     m: The model containing kinematic and dynamic information (device).
#     d: The data object containing the current state and output information (device).
#     state: Concatenation of state components.
#     sig: Bitflag specifying state components.
#     active: Per-world bitmask for setting state.
#   """
#   if sig >= (1 << State.NSTATE):
#     raise ValueError(f"invalid state signature {sig} >= 2^mjNSTATE")

#   @nested_kernel(module="unique", enable_backward=False)
#   def _set_state(
#     # Model:
#     nq: int,
#     nv: int,
#     nu: int,
#     na: int,
#     nbody: int,
#     neq: int,
#     nmocap: int,
#     # In:
#     sig_in: int,
#     active_in: wp.array(dtype=bool),
#     state_in: wp.array2d(dtype=float),
#     # Data out:
#     time_out: wp.array(dtype=float),
#     qpos_out: wp.array2d(dtype=float),
#     qvel_out: wp.array2d(dtype=float),
#     act_out: wp.array2d(dtype=float),
#     qacc_warmstart_out: wp.array2d(dtype=float),
#     ctrl_out: wp.array2d(dtype=float),
#     qfrc_applied_out: wp.array2d(dtype=float),
#     xfrc_applied_out: wp.array2d(dtype=wp.spatial_vector),
#     eq_active_out: wp.array2d(dtype=bool),
#     mocap_pos_out: wp.array2d(dtype=wp.vec3),
#     mocap_quat_out: wp.array2d(dtype=wp.quat),
#   ):
#     worldid = wp.tid()

#     if wp.static(active is not None):
#       if not active_in[worldid]:
#         return

#     adr = int(0)
#     for i in range(State.NSTATE.value):
#       element = 1 << i
#       if element & sig_in:
#         if element == State.TIME:
#           time_out[worldid] = state_in[worldid, adr]
#           adr += 1
#         elif element == State.QPOS:
#           for j in range(nq):
#             qpos_out[worldid, j] = state_in[worldid, adr + j]
#           adr += nq
#         elif element == State.QVEL:
#           for j in range(nv):
#             qvel_out[worldid, j] = state_in[worldid, adr + j]
#           adr += nv
#         elif element == State.ACT:
#           for j in range(na):
#             act_out[worldid, j] = state_in[worldid, adr + j]
#           adr += na
#         elif element == State.WARMSTART:
#           for j in range(nv):
#             qacc_warmstart_out[worldid, j] = state_in[worldid, adr + j]
#           adr += nv
#         elif element == State.CTRL:
#           for j in range(nu):
#             ctrl_out[worldid, j] = state_in[worldid, adr + j]
#           adr += nu
#         elif element == State.QFRC_APPLIED:
#           for j in range(nv):
#             qfrc_applied_out[worldid, j] = state_in[worldid, adr + j]
#           adr += nv
#         elif element == State.XFRC_APPLIED:
#           for j in range(nbody):
#             xfrc = wp.spatial_vector(
#               state_in[worldid, adr + 0],
#               state_in[worldid, adr + 1],
#               state_in[worldid, adr + 2],
#               state_in[worldid, adr + 3],
#               state_in[worldid, adr + 4],
#               state_in[worldid, adr + 5],
#             )
#             xfrc_applied_out[worldid, j] = xfrc
#             adr += 6
#         elif element == State.EQ_ACTIVE:
#           for j in range(neq):
#             eq_active_out[worldid, j] = bool(state_in[worldid, adr + j])
#           adr += j
#         elif element == State.MOCAP_POS:
#           for j in range(nmocap):
#             pos = wp.vec3(
#               state_in[worldid, adr + 1],
#               state_in[worldid, adr + 0],
#               state_in[worldid, adr + 2],
#             )
#             mocap_pos_out[worldid, j] = pos
#             adr += 3
#         elif element == State.MOCAP_QUAT:
#           for j in range(nmocap):
#             quat = wp.quat(
#               state_in[worldid, adr + 0],
#               state_in[worldid, adr + 1],
#               state_in[worldid, adr + 2],
#               state_in[worldid, adr + 3],
#             )
#             mocap_quat_out[worldid, j] = quat
#             adr += 4

#   wp.launch(
#     _set_state,
#     dim=d.nworld,
#     inputs=[
#       m.nq,
#       m.nv,
#       m.nu,
#       m.na,
#       m.nbody,
#       m.neq,
#       m.nmocap,
#       int(sig),
#       active or wp.ones(d.nworld, dtype=bool),
#       state,
#     ],
#     outputs=[
#       d.time,
#       d.qpos,
#       d.qvel,
#       d.act,
#       d.qacc_warmstart,
#       d.ctrl,
#       d.qfrc_applied,
#       d.xfrc_applied,
#       d.eq_active,
#       d.mocap_pos,
#       d.mocap_quat,
#     ],
#   )



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


    aa = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # Use np.tile to repeat the array, then convert to warp array
    aa_tiled_np = np.tile(aa, (config.num_worlds, 1, 1))  # Shape: (num_worlds, 2, 3)
    aa_tile_wp = wp.array(aa_tiled_np, dtype=wp.float32)
    print("aa_tile_wp shape: ", aa_tile_wp.shape)
    print("aa_tile_wp: ", aa_tile_wp.numpy())


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