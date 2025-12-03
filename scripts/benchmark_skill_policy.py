from performance_benchmark import benchmark_multiple_tasks_and_optimizers
import onnxruntime
from pathlib import Path

def benchmark_skill_policy(num_episodes: int = 50):
    optimizer_names = [
        "skill_policy",
    ]
    onnx_session_dict = None

    # Path to the ONNX model (relative to scripts directory)
    script_dir = Path(__file__).parent
    onnx_path = script_dir / "skill_policies" / "best_skill_policy (1).onnx"

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at: {onnx_path}\n"
            "Please ensure the skill policy model is placed at scripts/skill_policies/best_skill_policy.onnx"
        )

    print(f"Loading ONNX model from: {onnx_path}")
    session = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )
    print(f"✓ ONNX model loaded successfully")

    # Map the session to task names
    # NOTE: Skill policy requires tasks with nu=19 (full joint control)
    # Use spot_baseline tasks, not spot tasks with custom control mapping
    onnx_session_dict = {
        "spot_box_push_baseline": session,
    }

    # Run benchmark
    benchmark_multiple_tasks_and_optimizers(
        task_names=[
            "spot_box_push_baseline",  # Use spot_baseline tasks for skill policy (nu=19)
        ],
        optimizer_names=optimizer_names,
        num_episodes=num_episodes,
        episode_length_s=30.0,
        viz_dt=0.02,
        onnx_session_dict=onnx_session_dict,
        save_results=True,
    )

if __name__ == "__main__":
    benchmark_skill_policy(num_episodes=10)