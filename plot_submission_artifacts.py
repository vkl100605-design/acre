"""
Plot training evidence for hackathon / judges: Q-learning is produced by
`train_clean.py` (reward_curve.png, latency_curve.png). This script adds charts
from LLM+PPO `llm_training_log.jsonl` (JSONL) when present.
Run: python plot_submission_artifacts.py [--llm-log PATH] [--out-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _safe_floats(rows: List[Dict[str, Any]], key: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
        else:
            out.append(None)
    return out


def _find_loss_series(rows: List[Dict[str, Any]]) -> Tuple[str, List[Optional[float]]] | None:
    if not rows:
        return None
    sample = rows[0]
    for k, v in sample.items():
        if not k.startswith("stat_"):
            continue
        kl = k.lower()
        if "loss" in kl and "total" in kl and isinstance(v, (int, float)):
            return k, _safe_floats(rows, k)
    for k, v in sample.items():
        if k.startswith("stat_") and "loss" in k.lower() and isinstance(v, (int, float)):
            return k, _safe_floats(rows, k)
    return None


def _plot_series(
    xs: List[int],
    ys: List[float],
    title: str,
    ylabel: str,
    out: Path,
    color: str,
) -> bool:
    if not xs or not ys or len(xs) != len(ys):
        return False
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(xs, ys, color=color, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Step (log row)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def plot_llm_log_to_pngs(
    log_path: Path,
    out_dir: Path,
) -> Dict[str, str]:
    """
    Writes llm_reward_curve.png, llm_shaped_reward_curve.png, lmm_loss_curve.png (if loss in log).
    Returns paths that were created.
    out_dir: e.g. artifacts/llm_ppo/
    """
    rows = _load_jsonl(log_path)
    created: Dict[str, str] = {}
    if not rows:
        return created

    xs = list(range(1, len(rows) + 1))
    env = [float(r.get("env_reward", 0.0) or 0.0) for r in rows]
    shaped = [float(r.get("shaped_reward", 0.0) or 0.0) for r in rows]

    p1 = out_dir / "llm_env_reward_curve.png"
    if _plot_series(xs, env, "LLM+PPO: environment reward (per step)", "env_reward", p1, "#2563eb"):
        created["env_reward"] = str(p1)

    p2 = out_dir / "llm_shaped_reward_curve.png"
    if _plot_series(xs, shaped, "LLM+PPO: shaped reward (per step)", "shaped_reward", p2, "#059669"):
        created["shaped_reward"] = str(p2)

    loss_info = _find_loss_series(rows)
    if loss_info:
        k, vals = loss_info
        fvals: List[float] = []
        for v in vals:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                fvals.append(0.0)  # skip gaps for simple line; prefer showing curve
            else:
                fvals.append(float(v))
        p3 = out_dir / "llm_loss_curve.png"
        if _plot_series(
            xs,
            fvals,
            f"LLM+PPO: {k.replace('stat_', '')}",
            "loss",
            p3,
            "#dc2626",
        ):
            created["loss"] = str(p3)

    return created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-log",
        type=Path,
        default=Path("artifacts/llm_ppo/llm_training_log.jsonl"),
        help="Path to llm_training_log.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/llm_ppo"),
        help="Directory to write PNGs (same as log by default).",
    )
    args = parser.parse_args()
    out = plot_llm_log_to_pngs(args.llm_log, args.out_dir)
    print(json.dumps({"wrote": out}, indent=2))


if __name__ == "__main__":
    main()
