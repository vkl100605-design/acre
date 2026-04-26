from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt


def _base_figure(title: str, y_label: str):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, linestyle="--")
    return fig, ax


def _moving_average(values: List[float], window: int = 8) -> List[float]:
    if not values:
        return []
    if len(values) < window:
        return values[:]
    out: List[float] = []
    rolling = 0.0
    for idx, value in enumerate(values):
        rolling += value
        if idx >= window:
            rolling -= values[idx - window]
        denom = window if idx >= window - 1 else idx + 1
        out.append(rolling / denom)
    return out


def plot_reward_history(history: List[Dict[str, Any]]):
    fig, ax = _base_figure("Reward vs Episodes", "Reward")
    episodes = [item["episode"] for item in history]
    rewards = [item["reward"] for item in history]
    ax.plot(episodes, rewards, color="#2563eb", linewidth=2.0)
    ax.plot(episodes, _moving_average(rewards), color="#1d4ed8", linewidth=2.5, linestyle="--", alpha=0.85, label="Moving avg")
    ax.legend(loc="best")
    return fig


def plot_cost_history(history: List[Dict[str, Any]]):
    fig, ax = _base_figure("Cost vs Episodes", "Cost")
    episodes = [item["episode"] for item in history]
    costs = [item["cost"] for item in history]
    ax.plot(episodes, costs, color="#f97316", linewidth=2.0)
    ax.plot(episodes, _moving_average(costs), color="#ea580c", linewidth=2.5, linestyle="--", alpha=0.85, label="Moving avg")
    ax.legend(loc="best")
    return fig


def plot_uptime_history(history: List[Dict[str, Any]]):
    fig, ax = _base_figure("Uptime vs Episodes", "Uptime (%)")
    episodes = [item["episode"] for item in history]
    uptime = [item["uptime"] for item in history]
    ax.plot(episodes, uptime, color="#16a34a", linewidth=2.0)
    ax.plot(episodes, _moving_average(uptime), color="#15803d", linewidth=2.5, linestyle="--", alpha=0.85, label="Moving avg")
    ax.set_ylim(0, 100)
    ax.legend(loc="best")
    return fig


def plot_latency_history(history: List[Dict[str, Any]]):
    fig, ax = _base_figure("Latency vs Episodes", "Avg latency (ms / step)")
    episodes = [item["episode"] for item in history]
    latencies = [float(item.get("avg_latency_ms", 0.0)) for item in history]
    ax.plot(episodes, latencies, color="#a855f7", linewidth=2.0)
    ax.plot(
        episodes,
        _moving_average(latencies),
        color="#7c3aed",
        linewidth=2.5,
        linestyle="--",
        alpha=0.85,
        label="Moving avg",
    )
    ax.legend(loc="best")
    return fig


def _min_max_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = max(1e-9, hi - lo)
    return [(v - lo) / span for v in values]


def plot_cost_latency_uptime_combo(history: List[Dict[str, Any]]):
    fig, ax_left = plt.subplots(figsize=(10, 3.4))
    ax_left.set_title("Trade-off view (normalized per series)")
    ax_left.set_xlabel("Episode")
    ax_left.grid(alpha=0.25, linestyle="--")

    episodes = [item["episode"] for item in history]
    costs = [float(item.get("cost", 0.0)) for item in history]
    latencies = [float(item.get("avg_latency_ms", 0.0)) for item in history]
    uptimes = [float(item.get("uptime", 0.0)) / 100.0 for item in history]

    ax_left.plot(episodes, _min_max_norm(costs), color="#f97316", linewidth=2.0, label="Cost (norm)")
    ax_left.plot(episodes, _min_max_norm(latencies), color="#a855f7", linewidth=2.0, label="Latency (norm)")
    ax_left.set_ylabel("Normalized cost / latency")

    ax_right = ax_left.twinx()
    ax_right.plot(episodes, uptimes, color="#16a34a", linewidth=2.0, linestyle="--", label="Uptime (0-1)")
    ax_right.set_ylabel("Uptime (fraction)")
    ax_right.set_ylim(0.0, 1.05)

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")
    fig.tight_layout()
    return fig


def plot_before_after(random_eval: Dict[str, float], trained_eval: Dict[str, float]):
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
    metrics = [
        ("Reward", random_eval["reward"], trained_eval["reward"]),
        ("Cost", random_eval["cost"], trained_eval["cost"]),
        ("Uptime %", random_eval["uptime"], trained_eval["uptime"]),
        (
            "Avg latency (ms)",
            float(random_eval.get("avg_latency_ms", 0.0)),
            float(trained_eval.get("avg_latency_ms", 0.0)),
        ),
    ]
    for idx, (name, random_value, trained_value) in enumerate(metrics):
        axes[idx].bar(["Random", "Trained"], [random_value, trained_value], color=["#94a3b8", "#2563eb"])
        axes[idx].set_title(name)
        axes[idx].grid(axis="y", alpha=0.2, linestyle="--")
    fig.suptitle("Before vs After: Random vs Trained RL")
    fig.tight_layout()
    return fig
