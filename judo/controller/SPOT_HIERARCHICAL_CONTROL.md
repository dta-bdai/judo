# Spot Hierarchical Control Implementation Guide

This guide explains how to use the hierarchical locomotion controller for Spot tasks in Judo.

## Overview

Spot tasks require hierarchical control where:
1. **High-level MPC** optimizes commands (base velocity, arm positions, etc.) at ~10Hz
2. **Low-level locomotion policy** converts commands to joint targets at ~56Hz
3. **PD controller** converts joint targets to torques
4. **Physics simulation** steps at ~2000Hz with these torques

## Components

### 1. BatchedSpotLocomotionController (`judo/controller/spot_locomotion.py`)

GPU-accelerated locomotion policy that:
- Loads a PyTorch `.pt` model
- Computes batched observations from states
- Runs the policy to get actions
- Applies PD control to get torques

```python
from judo.controller.spot_locomotion import BatchedSpotLocomotionController

# Initialize controller
controller = BatchedSpotLocomotionController(
    model_path="path/to/locomotion_policy.pt",
    device="cuda:0",
    action_scale=0.2,
)

# Compute observations (batch_size, 84)
obs = controller.get_observation(qpos, qvel, previous_actions, cmd)

# Get target joint positions (batch_size, 19)
actions, target_q = controller.get_joint_target(obs, cmd)

# Compute PD torques (batch_size, 19)
torques = controller.compute_pd_torques(target_q, q, dq)
```

### 2. MJSimulatorLLC (`judo/utils/spot_hierarchical_rollout.py`)

Wrapper for performing single-step and trajectory rollouts:

```python
from judo.utils.spot_hierarchical_rollout import MJSimulatorLLC

# Initialize
hierarchical_rollout = MJSimulatorLLC(
    model=mujoco_model,
    locomotion_policy_path="path/to/policy.pt",
    physics_dt=0.0005,
    policy_frequency=56.0,
)

# Roll out one step
final_state, final_sensors = hierarchical_rollout.step(
    data=mujoco_data,
    high_level_cmd=cmd,  # (25,) array
    num_substeps=2,
)

# Roll out trajectory
states, sensors = hierarchical_rollout.rollout_trajectory(
    data=mujoco_data,
    high_level_cmds=cmds,  # (T, 25) array
    num_substeps=2,
)
```

### 3. HierarchicalRolloutBackend (`judo/utils/hierarchical_rollout_backend.py`)

GPU-batched rollout backend for use with BatchedControllers:

```python
from judo.utils.hierarchical_rollout_backend import HierarchicalRolloutBackend
from judo.controller import BatchedControllers

# Create hierarchical backend
backend = HierarchicalRolloutBackend(
    model=task.model,
    num_threads=64,
    num_problems=3,
    locomotion_policy_path="path/to/policy.pt",
)

# Use with BatchedControllers
batched_controllers = BatchedControllers(
    controller_config=controller_cfg,
    task=task,
    optimizer=optimizer,
    rollout_backend=backend,
)
```

## Usage in run_mpc.py

To use hierarchical control in `run_mpc.py`, you need to modify the simulation stepping:

### Before (direct MuJoCo stepping):
```python
# Advance physics for all simulations
for sim in sims:
    sim.step()
```

### After (with hierarchical control):
```python
from judo.utils.spot_hierarchical_rollout import MJSimulatorLLC

# Initialize hierarchical rollout (once, outside the loop)
hierarchical_rollouts = []
for sim in sims:
    if hasattr(sim.task, 'default_backend') and sim.task.default_backend == "mujoco_spot":
        hr = MJSimulatorLLC(
            model=sim.task.model,
            locomotion_policy_path="path/to/locomotion_policy.pt",
            physics_dt=sim.task.model.opt.timestep,
            policy_frequency=56.0,
        )
        hierarchical_rollouts.append(hr)
    else:
        hierarchical_rollouts.append(None)

# In the stepping loop:
for i, sim in enumerate(sims):
    if hierarchical_rollouts[i] is not None:
        # Spot task: use hierarchical control
        high_level_cmd = sim.control(sim.task.data.time)
        state, sensors = hierarchical_rollouts[i].step(
            data=sim.task.data,
            high_level_cmd=high_level_cmd,
            num_substeps=2,  # or sim.task.physics_substeps if available
        )
        # Update sim state
        sim.task.data.qpos[:] = state[:sim.task.model.nq]
        sim.task.data.qvel[:] = state[sim.task.model.nq:]
        sim.task.data.time += sim.timestep
    else:
        # Non-Spot task: use standard stepping
        sim.step()
```

## Locomotion Policy File

The locomotion policy should be a PyTorch state dict (`.pt` file) containing the weights for a 3-layer MLP:
- Input: 84 dimensions (observation)
- Hidden layers: 512 → 256 → 128 with ELU activation
- Output: 12 dimensions (leg actions)

Example policy paths:
- Starfish: `/home/dta_theaiinstitute_com/starfish/dexterity/data/policies/xinghao_policy_v1.onnx` (ONNX format - needs conversion)
- Judo: `judo/data/policies/spot_locomotion_policy.pt` (PyTorch format)

## Converting ONNX to PyTorch

If you have an ONNX model, convert it to PyTorch:

```python
import onnx
import torch
import torch.nn as nn
from onnx2pytorch import ConvertModel

# Load ONNX model
onnx_model = onnx.load("locomotion_policy.onnx")

# Convert to PyTorch
pytorch_model = ConvertModel(onnx_model)

# Extract the actor network (assuming it's the first module)
# You may need to adjust this based on your ONNX model structure
actor_state_dict = pytorch_model.state_dict()

# Save as PyTorch state dict
torch.save(actor_state_dict, "locomotion_policy.pt")
```

## Performance Considerations

### GPU Memory
- Each rollout backend maintains state for `num_threads * num_problems` parallel worlds
- Locomotion policy adds ~10MB GPU memory per backend
- Total GPU memory: ~100MB base + 10MB per backend

### Computational Cost
- Locomotion policy: ~0.5ms per batch of 64 rollouts
- PD control: ~0.1ms per batch
- Physics stepping: ~5ms per rollout step (GPU)

### Optimization Tips
1. **Use larger batches**: More parallel rollouts = better GPU utilization
2. **Adjust policy frequency**: Lower frequency (e.g., 28Hz instead of 56Hz) reduces overhead
3. **Use FP16**: Convert policy to half precision for 2x speedup
4. **Profile your code**: Use `torch.cuda.profiler` to identify bottlenecks

## Troubleshooting

### Issue: "No module named 'torch'"
Solution: Install PyTorch with CUDA support
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "CUDA out of memory"
Solutions:
- Reduce `num_threads` or `num_problems`
- Use CPU backend: `device="cpu"` (slower)
- Enable gradient checkpointing (if training)

### Issue: "Policy outputs are NaN"
Solutions:
- Check input normalization (observations should be properly scaled)
- Verify policy was trained with correct observation format
- Check for numerical instability in quaternion operations

### Issue: "Robot falls immediately"
Solutions:
- Verify PD gains (kp, kd) match training values
- Check torque limits are correct
- Ensure initial state is valid (standing pose)
- Verify command format (25-dim: base_vel(3) + arm(7) + legs(12) + torso(3))

## References

- Original implementation: `/home/dta_theaiinstitute_com/starfish/dexterity/`
- Policy training: Ask John for details on the RL locomotion training
- Hardware deployment: See `starfish/dexterity/hardware/` for real robot code
