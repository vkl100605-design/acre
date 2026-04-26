from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_clean import train_llm_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO+TRL training for SRE-LLM policy (ACRE++).")
    parser.add_argument("--episodes", type=int, default=4, help="outer loop episodes (PPO environment rollouts).")
    parser.add_argument("--task-id", default="incident_recovery")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model id; alternatives: TinyLlama/TinyLlama-1.1B-Chat-v1.0, microsoft/phi-2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/llm_ppo", help="saves model, tokenizer, logs.")
    args = parser.parse_args()

    result = train_llm_agent(
        task_id=args.task_id,
        episodes=int(args.episodes),
        seed=int(args.seed),
        model_name=str(args.model_name),
        output_dir=str(args.output_dir),
    )
    out = Path(args.output_dir)
    log = out / "llm_training_log.jsonl"
    if log.is_file() and int(log.stat().st_size) > 0:
        try:
            from plot_submission_artifacts import plot_llm_log_to_pngs

            plot_llm_log_to_pngs(log, out)
        except Exception:  # noqa: BLE001 — plot is optional, training already succeeded
            pass
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
