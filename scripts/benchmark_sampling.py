from performance_benchmark import benchmark_multiple_tasks_and_optimizers

def benchmark_sampling_pushing(num_episodes: int = 50):
    optimizer_names = [
        "cem",
    ]
    benchmark_multiple_tasks_and_optimizers(
        task_names=[
            # "spot_box_push",
            "spot_chair_push",
            "spot_rack_push",
            "spot_cone_push",
            "spot_tire_push",
        ],
        optimizer_names=optimizer_names,
        num_episodes=num_episodes,
        episode_length_s=30.0,
        viz_dt=0.02,
        save_results=False,
    )

if __name__ == "__main__":
    benchmark_sampling_pushing(num_episodes=10)