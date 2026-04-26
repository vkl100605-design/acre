from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from cost_agent import RuleBasedCostAgent
from evaluation import evaluate_policy, evaluate_random
from env import ACREEnv
from latency_agent import LatencyAgent
from rl_agent import QLearningAgent, save_agent
from tasks import get_task


ARTIFACT_DIR = Path("artifacts")


def _plot_series(values: List[float], title: str, ylabel: str, out_path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(list(range(1, len(values) + 1)), values, color=color, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def train_clean(
    *,
    task_id: str,
    episodes: int,
    seed: int,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.97,
) -> Dict[str, object]:
    task = get_task(task_id)
    agent = QLearningAgent(
        num_states=ACREEnv().observation_space_n,
        num_actions=5,
        alpha=alpha,
        gamma=gamma,
    )
    cost_agent = RuleBasedCostAgent()
    latency_agent = LatencyAgent()
    rng = random.Random(seed)

    reward_history: List[float] = []
    latency_history: List[float] = []
    cost_history: List[float] = []

    for episode in range(1, episodes + 1):
        env = ACREEnv(episode_length=task.episode_length, seed=seed + episode)
        state, info = env.reset(seed=seed + episode)
        done = False

        total_reward = 0.0
        total_cost = 0.0
        total_latency = 0.0
        steps = 0
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** (episode - 1)))

        # Explicit environment interaction loop (OpenEnv-style flow).
        while not done:
            action = agent.act(state, rng, epsilon=epsilon)
            cost_action = cost_agent.suggest_action(info)
            latency_action = latency_agent.suggest_action(info)
            next_state, reward, terminated, truncated, next_info = env.step(
                action,
                cost_action=cost_action,
                latency_action=latency_action,
            )
            done = bool(terminated or truncated)
            agent.update(state, action, float(reward), int(next_state), done)

            state = int(next_state)
            info = next_info
            total_reward += float(reward)
            total_cost += float(next_info.get("step_cost", 0.0))
            total_latency += float(next_info.get("latency_ms", 0.0))
            steps += 1

        reward_history.append(total_reward)
        cost_history.append(total_cost)
        latency_history.append(total_latency / max(1, steps))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACT_DIR / "q_table_clean.json"
    save_agent(
        model_path,
        task_id,
        agent,
        {
            "episodes": episodes,
            "seed": seed,
            "task_id": task_id,
            "alpha": alpha,
            "gamma": gamma,
        },
    )

    _plot_series(
        reward_history,
        "Reward vs Episodes",
        "Reward",
        ARTIFACT_DIR / "reward_curve.png",
        color="#2563eb",
    )
    _plot_series(
        latency_history,
        "Latency vs Episodes",
        "Avg Latency (ms)",
        ARTIFACT_DIR / "latency_curve.png",
        color="#a855f7",
    )

    trained_eval = evaluate_policy(task, seed=seed + 10_000, episodes=10, q_agent=agent)
    random_eval = evaluate_random(task, seed=seed + 20_000, episodes=10)
    reward_improvement = float(trained_eval["reward"]) - float(random_eval["reward"])
    cost_reduction = float(random_eval["cost"]) - float(trained_eval["cost"])
    uptime_improvement = float(trained_eval["uptime"]) - float(random_eval["uptime"])
    latency_improvement_ms = float(random_eval["avg_latency_ms"]) - float(trained_eval["avg_latency_ms"])

    summary = {
        "task_id": task_id,
        "episodes": episodes,
        "seed": seed,
        "avg_reward": sum(reward_history) / max(1, len(reward_history)),
        "avg_cost": sum(cost_history) / max(1, len(cost_history)),
        "avg_latency_ms": sum(latency_history) / max(1, len(latency_history)),
        "reward_curve_png": str(ARTIFACT_DIR / "reward_curve.png"),
        "latency_curve_png": str(ARTIFACT_DIR / "latency_curve.png"),
        "model_path": str(model_path),
        "trained_eval": trained_eval,
        "random_eval": random_eval,
        "reward_improvement": reward_improvement,
        "reward_improvement_pct": (reward_improvement / max(1e-9, abs(float(random_eval["reward"])))) * 100.0,
        "cost_reduction": cost_reduction,
        "uptime_improvement": uptime_improvement,
        "latency_improvement_ms": latency_improvement_ms,
        "baseline_text": (
            f"Random -> Reward: {random_eval['reward']:.2f} | Cost: {random_eval['cost']:.2f} | "
            f"Uptime: {random_eval['uptime']:.1f}% | Latency: {random_eval['avg_latency_ms']:.1f} ms\n"
            f"Trained -> Reward: {trained_eval['reward']:.2f} | Cost: {trained_eval['cost']:.2f} | "
            f"Uptime: {trained_eval['uptime']:.1f}% | Latency: {trained_eval['avg_latency_ms']:.1f} ms"
        ),
    }
    (ARTIFACT_DIR / "training_clean_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_llm_agent(
    *,
    task_id: str = "incident_recovery",
    episodes: int = 4,
    seed: int = 42,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "artifacts/llm_ppo",
) -> Dict[str, object]:
    """
    PPO (TRL) fine-tune on top of a small instruction LLM, using the same ACRE environment.
    Q-learning / train_clean() remains unchanged; call this for LLM+RL experiments.
    """
    from trl_training import run_ppo_training

    return run_ppo_training(
        task_id=task_id,
        episodes=episodes,
        model_name=model_name,
        seed=seed,
        output_dir=Path(output_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Q-learning training loop for OpenEnv compliance.")
    parser.add_argument("--task-id", default="incident_recovery")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_clean(task_id=args.task_id, episodes=args.episodes, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
