# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import time
from dataclasses import dataclass

import numpy as np
import tyro
from judo.app.structs import MujocoState
from judo.controller.controller import Controller
from judo.simulation.mj_simulation import MJSimulation
from judo.visualizers.visualizer import Visualizer
from tqdm import tqdm


@dataclass
class PublicMPCConfig:
    """Config class for MPC runs."""

    init_task: str = "cylinder_push"
    init_optimizer: str = "cem"
    visualize: bool = True
    steps: int = 1000


def update_visualization(visualizer: Visualizer, sim_state: MujocoState, traces: np.ndarray) -> None:
    visualizer.data.xpos[:] = sim_state.xpos
    visualizer.data.xquat[:] = sim_state.xquat
    visualizer.viser_model.set_data(visualizer.data)
    sensor_rollout_size = traces.shape[1]
    num_trace_sensors = traces.shape[2]
    all_traces_rollout_size = sensor_rollout_size * num_trace_sensors
    visualizer.viser_model.set_traces(traces, all_traces_rollout_size)


def run_mpc(config: PublicMPCConfig) -> None:
    """Sets up and runs a controller simulation with visualization."""
    controller = Controller(config.init_task, config.init_optimizer, rollout_backend="mujoco")
    simulation = MJSimulation(init_task=config.init_task)
    if config.visualize:
        visualizer = Visualizer(init_task=config.init_task, init_optimizer=config.init_optimizer)

    controller.reset()
    simulation.control = controller.action
    for _ in tqdm(range(config.steps), desc="Simulating..."):
        sim_state = simulation.sim_state
        controller.update_states(sim_state)
        controller.update_action()

        simulation.step()

        if config.visualize:
            update_visualization(visualizer, sim_state, controller.traces)
            time.sleep(simulation.timestep)


if __name__ == "__main__":
    run_mpc(tyro.cli(PublicMPCConfig))
    