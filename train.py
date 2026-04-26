from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from evaluation import save_training_artifacts, train_q_learning
from rl_agent import save_agent
from tasks import get_task


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "q_table.json"
TRAINING_HISTORY_PATH = ARTIFACT_DIR / "training_history.json"
TRAINING_SUMMARY_PATH = ARTIFACT_DIR / "training_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ACRE++ Q-learning policy.")
    parser.add_argument("--task-id", default="incident_recovery", choices=["incident_recovery", "stability_spike"])
    parser.add_argument("--episodes", type=int, default=100, help="Use 80-120 for final demo.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task = get_task(args.task_id)
    result = train_q_learning(task, episodes=args.episodes, seed=args.seed, alpha=0.1, gamma=0.9)

    save_training_artifacts(
        result,
        model_path=MODEL_PATH,
        history_path=TRAINING_HISTORY_PATH,
        summary_path=TRAINING_SUMMARY_PATH,
    )
    save_agent(
        MODEL_PATH,
        task.id,
        result["agent"],
        {
            "best_episode_reward": result["best_episode_reward"],
            "average_reward": result["average_reward"],
            "trained_eval": result["trained_eval"],
            "random_eval": result["random_eval"],
        },
    )

    print(f"[TRAIN] task={task.id} | episodes={args.episodes} | alpha=0.1 | gamma=0.9")
    print(
        f"Random -> Reward: {result['random_eval']['reward']:.2f} | Cost: {result['random_eval']['cost']:.2f} | "
        f"Uptime: {result['random_eval']['uptime']:.1f}% | "
        f"Avg latency: {result['random_eval'].get('avg_latency_ms', 0.0):.1f} ms"
    )
    print(
        f"Trained -> Reward: {result['trained_eval']['reward']:.2f} | Cost: {result['trained_eval']['cost']:.2f} | "
        f"Uptime: {result['trained_eval']['uptime']:.1f}% | "
        f"Avg latency: {result['trained_eval'].get('avg_latency_ms', 0.0):.1f} ms"
    )
    print(
        f"[SUMMARY] best_episode_reward={result['best_episode_reward']:.2f} | "
        f"average_reward={result['average_reward']:.2f}"
    )
    print(f"[ARTIFACTS] {MODEL_PATH}, {TRAINING_HISTORY_PATH}, {TRAINING_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
