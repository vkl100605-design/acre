from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from cost_agent import RuleBasedCostAgent
from env import ACTION_NAMES, ACREEnv
from latency_agent import LatencyAgent
from tasks import TASKS, TaskDefinition, get_task

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


ACTION_TO_INDEX = {
    "hold": 0,
    "scale_up": 1,
    "scale_down": 2,
    "restart": 3,
    "rollback": 4,
}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def sigmoid_score(value: float, scale: float = 10.0) -> float:
    return 1.0 / (1.0 + math.exp(-value / scale))


def traffic_level(value: float) -> str:
    if value < 0.4:
        return "low"
    if value < 0.7:
        return "medium"
    return "high"


def health_label(value: int) -> str:
    return {2: "healthy", 1: "degraded", 0: "down"}.get(int(value), "unknown")


def risk_label(value: int) -> str:
    return {0: "low", 1: "medium", 2: "high"}.get(int(value), "medium")


def build_observation_text(info: Dict) -> str:
    return (
        f"step={info.get('step', 0)}, traffic={traffic_level(float(info.get('traffic_demand', 0.0)))}, "
        f"load={float(info.get('load_ratio', 0.0)):.2f}x, health={health_label(int(info.get('health', 0)))}, "
        f"risk={risk_label(int(info.get('deployment_risk', 0)))}, cost={float(info.get('step_cost', 0.0)):.2f}"
    )


def heuristic_action(info: Dict) -> str:
    health = int(info.get("health", 2))
    load_ratio = float(info.get("load_ratio", 0.0))
    error_rate = float(info.get("error_rate", 0.0))
    risk = int(info.get("deployment_risk", 0))

    if health == 0 or error_rate > 0.55:
        return "rollback" if risk > 0 else "restart"
    if load_ratio > 1.25:
        return "scale_up"
    if load_ratio < 0.72 and int(info.get("capacity", 1)) > 1:
        return "scale_down"
    if health == 1:
        return "restart"
    return "hold"


class OpenAILLMAgent:
    def __init__(self, api_base_url: str, model_name: str, hf_token: str) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not available")
        self.client = OpenAI(base_url=api_base_url, api_key=hf_token)
        self.model_name = model_name

    def choose_action(self, task: TaskDefinition, info: Dict) -> Tuple[str, str]:
        prompt = (
            "You are controlling a cloud reliability agent.\n"
            "Choose exactly one action from: hold, scale_up, scale_down, restart, rollback.\n"
            "Return JSON with keys action and reason.\n"
            f"Task: {task.name}\n"
            f"Description: {task.description}\n"
            f"Observation: {build_observation_text(info)}\n"
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Respond with concise JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=80,
        )
        content = response.choices[0].message.content or ""
        parsed = extract_json(content)
        action = str(parsed.get("action", "")).strip().lower()
        reason = str(parsed.get("reason", "")).strip()
        if action not in ACTION_TO_INDEX:
            action = heuristic_action(info)
        if not reason:
            reason = "Fallback to heuristic or empty LLM response."
        return action, reason


def extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def normalize_score(total_reward: float) -> float:
    return clamp01(sigmoid_score(total_reward, scale=18.0))


def step_score(reward: float) -> float:
    return clamp01(sigmoid_score(reward, scale=4.0))


def print_start(task: TaskDefinition, model_name: str, api_base_url: str, use_llm: bool) -> None:
    print(
        f"[START] task_id={task.id} | task_name={task.name} | difficulty={task.difficulty} | "
        f"max_steps={task.max_steps} | model={model_name} | api_base_url={api_base_url} | use_llm={str(use_llm).lower()}"
    )


def print_step(task_id: str, step: int, action: str, reward: float, score: float, done: bool, note: str) -> None:
    print(
        f"[STEP] task_id={task_id} | step={step} | action={action} | reward={reward:+.4f} | "
        f"score={score:.4f} | done={str(done).lower()} | note={note}"
    )


def print_end(task_id: str, total_reward: float, score: float, steps: int, success: bool) -> None:
    print(
        f"[END] task_id={task_id} | steps={steps} | total_reward={total_reward:+.4f} | "
        f"score={score:.4f} | success={str(success).lower()}"
    )


def run_task(task: TaskDefinition, agent: Optional[OpenAILLMAgent], verbose: bool = False) -> Dict[str, float]:
    env = ACREEnv(episode_length=task.episode_length, seed=task.seed)
    cost_agent = RuleBasedCostAgent()
    latency_agent = LatencyAgent()
    state, info = env.reset(seed=task.seed)

    total_reward = 0.0
    total_score = 0.0
    llm_used = agent is not None

    print_start(task, model_name=os.getenv("MODEL_NAME", "heuristic"), api_base_url=os.getenv("API_BASE_URL", "none"), use_llm=llm_used)

    for step_idx in range(1, task.max_steps + 1):
        if agent is not None:
            action_name, reason = agent.choose_action(task, info)
        else:
            action_name = heuristic_action(info)
            reason = "Heuristic fallback."

        cost_action = cost_agent.suggest_action(info)
        latency_action = latency_agent.suggest_action(info)
        next_state, reward, terminated, truncated, info = env.step(
            ACTION_TO_INDEX[action_name],
            cost_action,
            latency_action=latency_action,
        )
        done = terminated or truncated

        score = step_score(reward)
        total_reward += reward
        total_score = normalize_score(total_reward)
        print_step(task.id, step_idx, action_name, reward, score, done, reason)

        state = next_state
        if done:
            break

    print_end(task.id, total_reward, total_score, step_idx, success=total_score >= 0.5)
    return {"task_id": task.id, "reward": total_reward, "score": total_score, "steps": step_idx}


def build_agent() -> Optional[OpenAILLMAgent]:
    api_base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not api_base_url or not model_name or not hf_token or OpenAI is None:
        return None
    return OpenAILLMAgent(api_base_url=api_base_url, model_name=model_name, hf_token=hf_token)


def main() -> None:
    parser = argparse.ArgumentParser(description="ACRE++ inference runner.")
    parser.add_argument("--tasks", nargs="*", default=[task.id for task in TASKS], help="Task IDs to run.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    agent = build_agent()
    results = []
    for task_id in args.tasks:
        task = get_task(task_id)
        results.append(run_task(task, agent, verbose=not args.quiet))

    aggregate_score = sum(r["score"] for r in results) / max(1, len(results))
    print(f"[END] task_id=aggregate | score={aggregate_score:.4f} | tasks={len(results)}")


if __name__ == "__main__":
    main()
