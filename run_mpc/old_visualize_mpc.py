# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

# gazelle:ignore viser

import json
import multiprocessing
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import tyro
from mujoco import mj_forward
from viser import GuiEvent, ViserServer

from judo.hardware.judo_app.io import JudoIOContext
from judo.tasks import get_registered_tasks
from judo.utils.json_serializer import decode_config
from judo.visualizers.mujoco.model import JudoMjModel


def visualize_trajectory_hdf5_dataset(dataset_path: Path) -> None:
    with h5py.File(str(dataset_path), "r") as f:
        policy_config_path = Path(f.attrs["config_path"])
        with open(str(policy_config_path), "r") as config_f:
            policy_config = json.load(config_f, object_hook=decode_config)

        qpos_dataset = f["qpos"]
        trajectory_length_dataset = f["trajectory_length"]

        visualize_trajectory_batch(policy_config["task"], qpos_dataset, trajectory_length_dataset)


def visualize_trajectory_batch(
    task: str, qpos_batch: np.ndarray | h5py.Dataset, trajectory_lengths: Optional[np.ndarray | h5py.Dataset] = None
) -> None:
    server = ViserServer()

    registered_tasks = get_registered_tasks()
    task_class, _ = registered_tasks[task]

    judo_context = JudoIOContext(multiprocessing.Manager())
    task_instance = task_class(judo_context)

    # Create judo object for Mujoco model.
    judo_mjmodel = JudoMjModel(server, task_instance.model)

    # Create GUI elements for controlling visualizer.
    running = False
    pause_button = server.gui.add_button("Start playback")

    def cycle_pause_button() -> None:
        nonlocal running
        if running:
            running = False
            pause_button.label = "Start playback"
        else:
            running = True
            pause_button.label = "Pause playback"

    @pause_button.on_click
    def _(_: GuiEvent) -> None:
        """More info about GUI callbacks in viser: https://viser.studio/versions/0.2.7/examples/03_gui_callbacks"""
        cycle_pause_button()

    trajectory_slider = server.gui.add_slider(
        "Trajectory index", min=0, max=len(qpos_batch) - 1, step=1, initial_value=0
    )
    if trajectory_lengths:
        curr_trajectory_length = trajectory_lengths[trajectory_slider.value] - 1
    else:
        curr_trajectory_length = len(qpos_batch[trajectory_slider.value]) - 1
    timestep_slider = server.gui.add_slider(
        "Trajectory timestep", min=0, max=curr_trajectory_length, step=1, initial_value=0
    )

    # Initialize visualization with the first state
    judo_mjmodel.data.qpos[:] = qpos_batch[0, 0]
    mj_forward(task_instance.model, judo_mjmodel.data)
    judo_mjmodel.set_data()

    @trajectory_slider.on_update
    def _(_: GuiEvent) -> None:
        """More info about GUI callbacks in viser: https://viser.studio/versions/0.2.7/examples/03_gui_callbacks"""
        # Reset timestep to zero for new trajectory.
        timestep_slider.value = 0
        # Change timestep bounds.
        if trajectory_lengths is not None:
            nonlocal curr_trajectory_length
            curr_trajectory_length = trajectory_lengths[trajectory_slider.value] - 1
            timestep_slider.max = curr_trajectory_length

    @timestep_slider.on_update
    def _(_: GuiEvent) -> None:
        """More info about GUI callbacks in viser: https://viser.studio/versions/0.2.7/examples/03_gui_callbacks"""
        # Update the judo_mjmodel's data object
        qpos_value = qpos_batch[trajectory_slider.value, timestep_slider.value]
        judo_mjmodel.data.qpos[:] = qpos_value
        mj_forward(task_instance.model, judo_mjmodel.data)

        judo_mjmodel.set_data()

    try:
        while True:
            if running:
                timestep_slider.value = min(curr_trajectory_length, timestep_slider.value + 1)
                if timestep_slider.value == curr_trajectory_length:
                    cycle_pause_button()

            time.sleep(task_instance.dt)
    except KeyboardInterrupt:
        print("Closing Judo...")


if __name__ == "__main__":
    tyro.cli(visualize_trajectory_hdf5_dataset)
