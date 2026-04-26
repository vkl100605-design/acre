from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from cost_agent import RuleBasedCostAgent
from env import ACTION_NAMES, ACREEnv
from latency_agent import LatencyAgent
from rl_agent import QLearningAgent
from tasks import TaskDefinition, get_task


ACTION_INDEX = {name: idx for idx, name in ACTION_NAMES.items()}


def action_name(action_idx: Optional[int]) -> str:
    if action_idx is None:
        # Agents that "stay quiet" on a step are treated as hold/none for UX readability.
        return "hold"
    return ACTION_NAMES.get(int(action_idx), "unknown")


def health_status(health: int) -> str:
    if int(health) >= 2:
        return "healthy"
    if int(health) == 1:
        return "degraded"
    return "down"


def explain_decision(info: Dict[str, Any], overridden: bool) -> str:
    load_ratio = float(info.get("load_ratio", 1.0))
    health = int(info.get("health", 2))
    final_action = int(info.get("final_action", 0))
    latency_ms = float(info.get("latency_ms", 0.0))
    override_agent = str(info.get("override_agent", "none"))
    if bool(info.get("overridden_by_latency_agent", False)) or override_agent == "latency":
        return (
            "Latency agent escalated to scale up because user experience risk was elevated "
            f"(latency ~{latency_ms:.0f} ms) while load sat around {load_ratio:.2f}x."
        )
    if overridden:
        return "Cost agent overrode RL to reduce spending pressure under current demand."
    if health == 0:
        return "Recovery-first action selected because the service is down."
    if load_ratio > 1.15 and final_action == ACTION_INDEX["scale_up"]:
        return "Scaling up protected reliability under high demand."
    if load_ratio < 0.75 and final_action == ACTION_INDEX["scale_down"]:
        return "Scaling down avoided unnecessary cost because demand is low."
    if latency_ms >= 220.0 and final_action == ACTION_INDEX["scale_up"]:
        return "Scaling up reduced end-user latency risk under elevated load or errors."
    if final_action in (ACTION_INDEX["restart"], ACTION_INDEX["rollback"]):
        return "Stability action selected to reduce risk and restore service health."
    return "Maintained current capacity to balance cost, reliability, and latency."


def timeline_status(info: Dict[str, Any], overridden: bool) -> str:
    health = int(info.get("health", 2))
    latency_ms = float(info.get("latency_ms", 0.0))
    if health == 0:
        return "❌ Failure"
    if bool(info.get("overridden_by_latency_agent", False)):
        return "⚡ Latency override"
    if overridden:
        return "⚠️ Cost override"
    if latency_ms >= 260.0:
        return "⚡ Latency spike"
    if health == 2:
        return "✅ Recovery" if int(info.get("step", 0)) > 1 else "OK"
    return "🟡 Degraded"


@dataclass
class EpisodeSummary:
    episode: int
    reward: float
    cost: float
    uptime: float
    avg_latency_ms: float
    overrides: int
    override_rate: float
    latency_overrides: int
    latency_override_rate: float
    steps: int
    epsilon: float


@dataclass
class StepTrace:
    step: int
    rl_action: str
    cost_action: str
    latency_action: str
    final_action: str
    overridden: bool
    overridden_by_cost_agent: bool
    overridden_by_latency_agent: bool
    override_agent: str
    conflict_type: str
    reward: float
    cost: float
    latency_ms: float
    uptime_state: str
    load_ratio: float
    reason: str
    status: str
    llm_reason: str = ""


def run_episode(
    task: TaskDefinition,
    *,
    seed: int,
    q_agent: Optional[QLearningAgent] = None,
    llm_agent: Optional[Any] = None,
    epsilon: float = 0.0,
    train: bool = False,
    random_policy: bool = False,
    collect_trace: bool = False,
) -> Tuple[EpisodeSummary, List[StepTrace]]:
    if q_agent is not None and llm_agent is not None:
        raise ValueError("Provide only one of q_agent or llm_agent")
    env = ACREEnv(episode_length=task.episode_length, seed=seed)
    cost_agent = RuleBasedCostAgent()
    latency_agent = LatencyAgent()
    rng = random.Random(seed)
    state, info = env.reset(seed=seed)

    total_reward = 0.0
    total_cost = 0.0
    total_latency_ms = 0.0
    healthy_steps = 0
    overrides = 0
    latency_overrides = 0
    trace: List[StepTrace] = []
    steps = 0

    for step in range(1, task.max_steps + 1):
        steps = step
        llm_reason = ""
        if random_policy or (q_agent is None and llm_agent is None):
            rl_action = rng.randrange(env.action_space_n)
        elif llm_agent is not None:
            act_name, llm_reason = llm_agent.choose_action(info)
            rl_action = int(ACTION_INDEX.get(str(act_name).strip().lower(), 0))
        else:
            rl_action = q_agent.act(state, rng, epsilon=epsilon)

        cost_action = cost_agent.suggest_action(info)
        latency_action = latency_agent.suggest_action(info)
        next_state, reward, terminated, truncated, next_info = env.step(
            rl_action,
            cost_action,
            latency_action=latency_action,
        )
        done = terminated or truncated

        if train and q_agent is not None and llm_agent is None:
            q_agent.update(state, rl_action, reward, next_state, done)

        total_reward += float(reward)
        total_cost += float(next_info.get("step_cost", 0.0))
        step_latency_ms = float(next_info.get("latency_ms", 0.0))
        total_latency_ms += step_latency_ms
        health = int(next_info.get("health", 0))
        healthy_steps += int(health == 2)
        overridden_cost = bool(next_info.get("overridden_by_cost_agent", False))
        overridden_latency = bool(next_info.get("overridden_by_latency_agent", False))
        overridden = overridden_cost or overridden_latency
        overrides += int(overridden_cost)
        latency_overrides += int(overridden_latency)

        if collect_trace:
            if llm_agent is not None:
                reason_text = (llm_reason or "").strip() or explain_decision(next_info, overridden_cost)
            else:
                reason_text = explain_decision(next_info, overridden_cost)
            trace.append(
                StepTrace(
                    step=step,
                    rl_action=action_name(rl_action),
                    cost_action=action_name(cost_action),
                    latency_action=action_name(latency_action),
                    final_action=action_name(int(next_info.get("final_action", rl_action))),
                    overridden=overridden,
                    overridden_by_cost_agent=overridden_cost,
                    overridden_by_latency_agent=overridden_latency,
                    override_agent=str(next_info.get("override_agent", "none")),
                    conflict_type=str(next_info.get("conflict_type", "none")),
                    reward=float(reward),
                    cost=float(next_info.get("step_cost", 0.0)),
                    latency_ms=step_latency_ms,
                    uptime_state=health_status(health),
                    load_ratio=float(next_info.get("load_ratio", 1.0)),
                    reason=reason_text,
                    status=timeline_status(next_info, overridden_cost),
                    llm_reason=llm_reason,
                )
            )

        state = next_state
        info = next_info
        if done:
            break

    uptime = 100.0 * healthy_steps / max(1, steps)
    avg_latency_ms = total_latency_ms / max(1, steps)
    summary = EpisodeSummary(
        episode=seed,
        reward=total_reward,
        cost=total_cost,
        uptime=uptime,
        avg_latency_ms=avg_latency_ms,
        overrides=overrides,
        override_rate=100.0 * overrides / max(1, steps),
        latency_overrides=latency_overrides,
        latency_override_rate=100.0 * latency_overrides / max(1, steps),
        steps=steps,
        epsilon=epsilon,
    )
    return summary, trace


def evaluate_policy(task: TaskDefinition, *, seed: int, episodes: int, q_agent: QLearningAgent) -> Dict[str, float]:
    return evaluate_common(task, seed=seed, episodes=episodes, q_agent=q_agent, random_policy=False)


def evaluate_random(task: TaskDefinition, *, seed: int, episodes: int) -> Dict[str, float]:
    return evaluate_common(task, seed=seed, episodes=episodes, q_agent=None, random_policy=True)


def evaluate_llm(task: TaskDefinition, *, seed: int, episodes: int, llm_agent: Any) -> Dict[str, float]:
    return evaluate_common(
        task,
        seed=seed,
        episodes=episodes,
        q_agent=None,
        random_policy=False,
        llm_agent=llm_agent,
    )


def evaluate_common(
    task: TaskDefinition,
    *,
    seed: int,
    episodes: int,
    q_agent: Optional[QLearningAgent],
    random_policy: bool,
    llm_agent: Optional[Any] = None,
) -> Dict[str, float]:
    rewards: List[float] = []
    costs: List[float] = []
    uptimes: List[float] = []
    latencies: List[float] = []
    overrides: List[float] = []
    latency_overrides: List[float] = []
    for idx in range(episodes):
        summary, _ = run_episode(
            task,
            seed=seed + idx,
            q_agent=q_agent,
            llm_agent=llm_agent,
            epsilon=0.0,
            train=False,
            random_policy=random_policy,
            collect_trace=False,
        )
        rewards.append(summary.reward)
        costs.append(summary.cost)
        uptimes.append(summary.uptime)
        latencies.append(summary.avg_latency_ms)
        overrides.append(summary.override_rate)
        latency_overrides.append(summary.latency_override_rate)
    return {
        "reward": sum(rewards) / max(1, len(rewards)),
        "cost": sum(costs) / max(1, len(costs)),
        "uptime": sum(uptimes) / max(1, len(uptimes)),
        "avg_latency_ms": sum(latencies) / max(1, len(latencies)),
        "override_rate": sum(overrides) / max(1, len(overrides)),
        "latency_override_rate": sum(latency_overrides) / max(1, len(latency_overrides)),
    }


def train_q_learning(
    task: TaskDefinition,
    *,
    episodes: int = 100,
    seed: int = 42,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.97,
) -> Dict[str, Any]:
    agent = QLearningAgent(num_states=ACREEnv().observation_space_n, num_actions=5, alpha=alpha, gamma=gamma)
    history: List[EpisodeSummary] = []
    best_episode_reward = float("-inf")

    for episode in range(1, episodes + 1):
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** (episode - 1)))
        summary, _ = run_episode(
            task,
            seed=seed + episode,
            q_agent=agent,
            epsilon=epsilon,
            train=True,
            random_policy=False,
            collect_trace=False,
        )
        summary.episode = episode
        history.append(summary)
        if summary.reward > best_episode_reward:
            best_episode_reward = summary.reward

    trained_eval = evaluate_policy(task, seed=seed + 10_000, episodes=10, q_agent=agent)
    random_eval = evaluate_random(task, seed=seed + 20_000, episodes=10)

    avg_reward = sum(item.reward for item in history) / max(1, len(history))
    reward_improvement = trained_eval["reward"] - random_eval["reward"]
    cost_reduction = random_eval["cost"] - trained_eval["cost"]
    uptime_improvement = trained_eval["uptime"] - random_eval["uptime"]
    latency_improvement = random_eval["avg_latency_ms"] - trained_eval["avg_latency_ms"]

    return {
        "task_id": task.id,
        "agent": agent,
        "history": history,
        "trained_eval": trained_eval,
        "random_eval": random_eval,
        "best_episode_reward": best_episode_reward,
        "average_reward": avg_reward,
        "reward_improvement": reward_improvement,
        "reward_improvement_pct": (reward_improvement / max(1e-9, abs(random_eval["reward"]))) * 100.0,
        "cost_reduction": cost_reduction,
        "uptime_improvement": uptime_improvement,
        "latency_improvement_ms": latency_improvement,
        "baseline_text": (
            f"Random -> Reward: {random_eval['reward']:.2f} | Cost: {random_eval['cost']:.2f} | "
            f"Uptime: {random_eval['uptime']:.1f}% | Latency: {random_eval['avg_latency_ms']:.1f} ms\n"
            f"Trained -> Reward: {trained_eval['reward']:.2f} | Cost: {trained_eval['cost']:.2f} | "
            f"Uptime: {trained_eval['uptime']:.1f}% | Latency: {trained_eval['avg_latency_ms']:.1f} ms"
        ),
    }


def save_training_artifacts(
    result: Dict[str, Any],
    *,
    model_path: Path,
    history_path: Path,
    summary_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_payload = [asdict(item) for item in result["history"]]
    summary_payload = {
        "task_id": result["task_id"],
        "best_episode_reward": result["best_episode_reward"],
        "average_reward": result["average_reward"],
        "reward_improvement": result["reward_improvement"],
        "reward_improvement_pct": result["reward_improvement_pct"],
        "cost_reduction": result["cost_reduction"],
        "uptime_improvement": result["uptime_improvement"],
        "latency_improvement_ms": result["latency_improvement_ms"],
        "trained_eval": result["trained_eval"],
        "random_eval": result["random_eval"],
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
