# Skill Policy Optimizer

The Skill Policy Optimizer is a network-based controller that can be used as an alternative to sample-based optimizers in the JUDO benchmark framework.

## Overview

Unlike traditional sample-based optimizers (CEM, MPPI, etc.) that sample and optimize trajectories, the Skill Policy Optimizer uses a pre-trained neural network to directly predict control actions from the current state.

### Key Differences from Sample-Based Optimizers

| Aspect | Sample-Based (CEM/MPPI) | Skill Policy |
|--------|------------------------|--------------|
| **Method** | Sample trajectories, evaluate, optimize | Direct network inference |
| **Iterations** | Multiple (e.g., 10-50) | Single forward pass |
| **Rollouts** | Many (e.g., 16-128) | One |
| **Speed** | Slower (many evaluations) | Faster (one inference) |
| **Flexibility** | Adapts to any task/reward | Task-specific (requires training) |

## Architecture

The Skill Policy Optimizer wraps an ONNX model and fits it into the `Optimizer` interface by treating it as a "degenerate optimizer":

1. **Sample phase**: Runs the network once to get predicted actions
2. **Update phase**: Returns the network output directly (no optimization)

### Expected ONNX Model Interface

The ONNX model should accept these inputs (all with batch dimension):

**Inputs:**
- `robot_pos_w`: Robot base position in world frame, shape `(batch, 3)`
- `robot_quat_w`: Robot base quaternion (w,x,y,z), shape `(batch, 4)`
- `robot_lin_vel_w`: Robot linear velocity in world frame, shape `(batch, 3)`
- `robot_ang_vel_w`: Robot angular velocity in world frame, shape `(batch, 3)`
- `joint_pos`: Joint positions (Isaac ordering), shape `(batch, 19)`
- `joint_vel`: Joint velocities (Isaac ordering), shape `(batch, 19)`
- `object_pos_w`: Object position in world frame, shape `(batch, 3)`
- `object_quat_w`: Object quaternion (w,x,y,z), shape `(batch, 4)`
- `object_lin_vel_w`: Object linear velocity, shape `(batch, 3)`
- `object_ang_vel_w`: Object angular velocity in world frame, shape `(batch, 3)`
- `target_cmd_w`: Target command (pos + quat), shape `(batch, 7)`
- `previous_skill_action`: Previous skill action, shape `(batch, skill_action_length)`
- `previous_arm_target_pos`: Previous arm target positions, shape `(batch, 7)`
- `previous_locomotion_action`: Previous locomotion action, shape `(batch, 12)`
- `locomotion_only`: Boolean flag, shape `(batch, 1)`

**Outputs:**
- `leg_pos_target`: Target leg joint positions (Isaac ordering), shape `(batch, 12)`
- `arm_target_pos`: Target arm joint positions, shape `(batch, 7)`
- `locomotion_actions`: Locomotion action vector, shape `(batch, 12)`
- `skill_actions`: Skill action vector, shape `(batch, skill_action_length)`

## Usage

### 1. Load ONNX Model

```python
import onnxruntime

session = onnxruntime.InferenceSession(
    "path/to/skill_policy.onnx",
    providers=["CPUExecutionProvider"]
)
```

### 2. Configure Skill Policy Optimizer

```python
from judo.optimizers import SkillPolicyConfig
import numpy as np

config = SkillPolicyConfig(
    onnx_session=session,
    target_cmd=np.array([1.0, 0.0, 0.3]),  # Target xyz position
    locomotion_only=False,
    num_rollouts=1,  # Always 1 for skill policy
    num_nodes=4,     # Determines trajectory horizon
    cutoff_time=0.2,
)
```

### 3. Use in Controller

```python
from judo.controller import Controller, ControllerConfig
from judo.optimizers import get_registered_optimizers

# Get the skill policy optimizer
optimizer_cls, _ = get_registered_optimizers()["skill_policy"]
optimizer = optimizer_cls(config, task.nu)

# Create controller with max_opt_iters=1 (no optimization loop needed)
controller_cfg = ControllerConfig(
    max_opt_iters=1,
    horizon=1.0,
)

controller = Controller(
    task=task,
    task_cfg=task_cfg,
    optimizer=optimizer,
    optimizer_cfg=config,
    controller_cfg=controller_cfg,
)

# Use normally
state = np.concatenate([data.qpos, data.qvel])
controller.update_action(state, current_time)
action = controller.action(current_time)
```

### 4. Benchmark Against Sample-Based Optimizers

```python
# Run skill policy
skill_times = []
for _ in range(100):
    start = time.perf_counter()
    controller_skill.update_action(state, 0.0)
    skill_times.append(time.perf_counter() - start)

# Run MPPI for comparison
mppi_times = []
for _ in range(100):
    start = time.perf_counter()
    controller_mppi.update_action(state, 0.0)
    mppi_times.append(time.perf_counter() - start)

print(f"Skill Policy: {np.mean(skill_times)*1000:.2f} ms")
print(f"MPPI: {np.mean(mppi_times)*1000:.2f} ms")
print(f"Speedup: {np.mean(mppi_times)/np.mean(skill_times):.2f}x")
```

## Configuration Options

### SkillPolicyConfig Parameters

- **onnx_session** (required): ONNX runtime InferenceSession
- **target_cmd**: Target position (xyz), default `np.zeros(3)`
- **locomotion_only**: Use only locomotion (no arm), default `False`
- **policy_frequency**: Policy update rate in Hz, default `50.0`
- **skill_action_length**: Length of skill action vector, default `10`
- **num_rollouts**: Always `1` (skill policy doesn't sample)
- **num_nodes**: Number of trajectory knot points (affects horizon)
- **cutoff_time**: Rollout cutoff time in seconds

## State Indexing

The optimizer automatically handles state indexing in two ways:

1. **With MuJoCo Model**: Uses `get_pos_indices` and `get_vel_indices` to dynamically extract state components
2. **Without Model**: Assumes standard 64-dim state layout (base + 19 joints + object)

The state is automatically converted from MuJoCo ordering to Isaac ordering using `mujoco_to_isaac()` before being passed to the network.

## Coordinate Frame Conversions

- **Angular Velocities**: Automatically converted from body frame to world frame using quaternion rotations
- **Joint Ordering**: Converted from MuJoCo ordering to Isaac ordering (and back)
- **Target Command**: Augmented with identity quaternion `[1,0,0,0]` if only position provided

## Example Scripts

See `scripts/benchmark_skill_policy_example.py` for complete examples including:
- Loading and configuring the skill policy
- Setting up benchmarks
- Comparing against sample-based optimizers

## Notes

- The skill policy requires `max_opt_iters=1` since no iterative optimization is performed
- Performance depends on ONNX model size and inference speed
- The policy is task-specific and must be trained for each task
- State extraction assumes standard Spot robot layout; custom tasks may need to override `convert_state_to_onnx_input()`

## Troubleshooting

### "onnx_session must be provided in SkillPolicyConfig"
Make sure to pass the loaded ONNX session when creating the config:
```python
config = SkillPolicyConfig(onnx_session=session, ...)
```

### "Must call set_current_state() before sampling"
This should be handled automatically by the controller. If you see this error, ensure you're using the updated controller code.

### State dimension mismatch
Check that your MuJoCo model matches the expected state layout. You may need to override `convert_state_to_onnx_input()` for custom tasks.

### ONNX input/output mismatch
Verify your ONNX model has the expected input/output names and shapes. Use:
```python
print([input.name for input in session.get_inputs()])
print([output.name for output in session.get_outputs()])
```
