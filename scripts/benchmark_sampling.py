from performance_benchmark import benchmark_multiple_tasks_and_optimizers

def benchmark_sampling_sumo(num_episodes: int = 50, save_results: bool = False):
    optimizer_names = [
        "cem",
    ]
    benchmark_multiple_tasks_and_optimizers(
        task_names=[
            "spot_box_push",
            "spot_chair_push",
            "spot_rack_push",
            "spot_cone_push",
            "spot_tire_push",
        ],
        optimizer_names=optimizer_names,
        num_episodes=num_episodes,
        episode_length_s=30.0,
        viz_dt=0.02,
        save_results=save_results,
    )

def benchmark_sampling_baseline(num_episodes: int = 50, save_results: bool = False):
    optimizer_names = [
        "cem",
    ]
    benchmark_multiple_tasks_and_optimizers(
        task_names=[
            "spot_box_push_baseline",
            # "spot_chair_push_baseline",
            # "spot_tire_push_baseline",
            # "spot_rack_push_baseline",
            # "spot_cone_push_baseline",
        ],
        optimizer_names=optimizer_names,
        num_episodes=num_episodes,
        episode_length_s=30.0,
        viz_dt=0.02,
        save_results=save_results,
    )

if __name__ == "__main__":
    # benchmark_sampling_sumo(num_episodes=50)
    benchmark_sampling_baseline(num_episodes=1, save_results=True)