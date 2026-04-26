from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from env import ACTION_NAMES, ACREEnv
from rl_agent import QLearningAgent, load_agent
from tasks import get_task


ARTIFACT_DIR = Path("artifacts")
ACTION_TO_TEXT = ACTION_NAMES


def _format_state(info: Dict) -> str:
    return (
        f"load={float(info.get('load_ratio', 0.0)):.2f}x, "
        f"health={int(info.get('health', 2))}, "
        f"error={float(info.get('error_rate', 0.0)):.2f}, "
        f"cost={float(info.get('step_cost', 0.0)):.2f}, "
        f"latency={float(info.get('latency_ms', 0.0)):.1f}ms"
    )


def _simple_policy(info: Dict) -> int:
    if float(info.get("latency_ms", 0.0)) > 220.0:
        return 1  # scale_up
    if float(info.get("load_ratio", 0.0)) > 1.15:
        return 1
    if float(info.get("load_ratio", 0.0)) < 0.72 and int(info.get("capacity", 1)) > 1:
        return 2  # scale_down
    return 0  # hold


def _q_agent_action(agent: Optional[QLearningAgent], state: int, info: Dict) -> int:
    if agent is not None:
        return int(agent.best_action(int(state)))
    return _simple_policy(info)


def collect_interaction_dataset(
    task_id: str = "incident_recovery",
    episodes: int = 4,
    model_path: Path = ARTIFACT_DIR / "q_table_clean.json",
) -> List[Dict]:
    task = get_task(task_id)
    rows: List[Dict] = []
    q_agent = load_agent(model_path)

    for idx in range(episodes):
        env = ACREEnv(episode_length=task.episode_length, seed=task.seed + idx)
        state, info = env.reset(seed=task.seed + idx)
        done = False

        while not done:
            action = _q_agent_action(q_agent, int(state), info)
            next_state, reward, terminated, truncated, next_info = env.step(action)
            rows.append(
                {
                    "task_id": task_id,
                    "state_text": _format_state(info),
                    "action": ACTION_TO_TEXT.get(action, "hold"),
                    "reward": float(reward),
                    "policy_source": "trained_q_agent" if q_agent is not None else "heuristic_fallback",
                    "next_state": int(next_state),
                }
            )
            done = bool(terminated or truncated)
            state = int(next_state)
            info = next_info
    return rows


def build_trl_compat_report(dataset: List[Dict]) -> Dict:
    trl_available = False
    unsloth_available = False
    ppo_config = None

    try:
        try:
            from trl import PPOConfig  # type: ignore
        except Exception:
            from trl.experimental.ppo import PPOConfig  # type: ignore

        try:
            cfg = PPOConfig(batch_size=4, mini_batch_size=1, learning_rate=1e-5)
        except (ValueError, TypeError, RuntimeError):
            # TRL 1.x PPOConfig inherits TrainingArguments: needs valid device flags on CPU
            cfg = PPOConfig(  # type: ignore[call-arg]
                output_dir=".",
                use_cpu=True,
                bf16=False,
                fp16=False,
                batch_size=4,
                mini_batch_size=1,
                learning_rate=1e-5,
            )
        ppo_config = {
            "batch_size": int(getattr(cfg, "batch_size", 4) or 4),
            "mini_batch_size": int(getattr(cfg, "mini_batch_size", 1) or 1),
            "learning_rate": float(getattr(cfg, "learning_rate", 1e-5)),
        }
        trl_available = True
    except Exception:
        ppo_config = {
            "batch_size": 4,
            "mini_batch_size": 1,
            "learning_rate": 1e-5,
            "note": "trl not installed; using compatibility stub",
        }

    try:
        import unsloth  # type: ignore  # noqa: F401

        unsloth_available = True
    except Exception:
        unsloth_available = False

    return {
        "trl_available": trl_available,
        "unsloth_available": unsloth_available,
        "dataset_rows": len(dataset),
        "sample": dataset[0] if dataset else {},
        "ppo_config": ppo_config,
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACT_DIR / "q_table_clean.json"
    dataset = collect_interaction_dataset(model_path=model_path)
    dataset_path = ARTIFACT_DIR / "trl_interactions.jsonl"
    with dataset_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")

    report = build_trl_compat_report(dataset)
    report["dataset_path"] = str(dataset_path)
    report["model_path"] = str(model_path)
    report["policy_source"] = (
        "trained_q_agent" if load_agent(model_path) is not None else "heuristic_fallback"
    )
    report_path = ARTIFACT_DIR / "trl_compat_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
