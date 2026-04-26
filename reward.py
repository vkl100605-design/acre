from __future__ import annotations

import re
from typing import Optional

_TRADEOFF_PATTERN = re.compile(
    r"(?i)(cost|latency|reliability|uptime|spend|budget|performance|slo|user experience|trade[- ]?off|balance|capacity)"
)

_GENERIC_PATTERN = re.compile(
    r"(?i)\b(as an ai|i cannot|generally|probably|i think|maybe)\b|^(ok|note|n/a)\b"
)


def reasoning_bonus(reasoning: str, *, chosen_action: str) -> float:
    r = (reasoning or "").strip().lower()

    # ❌ Too short = useless
    if len(r) < 20:
        return -0.3

    # ❌ Generic reasoning
    if _GENERIC_PATTERN.search(r) and not _TRADEOFF_PATTERN.search(r):
        return -0.2

    # ✅ Tradeoff detection
    hits = {m.group(0) for m in _TRADEOFF_PATTERN.finditer(r)}
    tradeoff_score = 0.4 if len(hits) >= 2 else 0.15 if len(hits) == 1 else 0.0

    # ❌ Action inconsistency
    consistency_penalty = 0.0
    if chosen_action == "scale_up" and "scale down" in r and "scale up" not in r:
        consistency_penalty += 0.3
    if chosen_action == "scale_down" and "scale up" in r and "scale down" not in r:
        consistency_penalty += 0.3

    # ❌ No reasoning for recovery actions
    if chosen_action in ("restart", "rollback"):
        if not any(k in r for k in ["recover", "failure", "risk", "health"]):
            consistency_penalty += 0.2

    score = tradeoff_score - consistency_penalty
    return max(-0.5, min(0.5, score))


def combined_step_reward(
    env_reward: float,
    reasoning: str,
    chosen_action: str,
    *,
    info: Optional[dict] = None,
    normalize: bool = True,
) -> float:

    info = info or {}

    # 🧠 Base
    reward = float(env_reward)

    # 🔥 1. Reasoning quality
    reward += reasoning_bonus(reasoning, chosen_action=chosen_action)

    # 🔥 2. Cost penalty (VERY IMPORTANT)
    step_cost = float(info.get("step_cost", 0.0))
    reward -= 0.15 * step_cost

    # 🔥 3. Uptime / health penalty
    health = int(info.get("health", 2))
    if health == 0:      # system down
        reward -= 3.0
    elif health == 1:    # degraded
        reward -= 1.0

    # 🔥 4. Anti "always scale_up"
    if chosen_action == "scale_up":
        reward -= 0.3

    # 🔥 5. Encourage recovery actions when needed
    if health < 2 and chosen_action in ("restart", "rollback"):
        reward += 0.5

    # 🔥 6. Clip for PPO stability
    if normalize:
        reward = max(-10.0, min(10.0, reward))

    return reward