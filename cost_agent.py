from __future__ import annotations

from typing import Dict, Optional


class RuleBasedCostAgent:
    """
    Lightweight FinOps-style agent.

    It does not learn. It just suggests cost-aware actions and sometimes
    overrides wasteful scaling choices when demand is low.
    """

    def suggest_action(self, info: Dict) -> int:
        health = int(info["health"])
        error_rate = float(info["error_rate"])
        load_ratio = float(info["load_ratio"])
        capacity = int(info["capacity"])
        traffic_demand = float(info["traffic_demand"])
        step_cost = float(info["step_cost"])
        max_capacity = int(info.get("max_capacity", 5))

        # If the system is failing, prioritize recovery first.
        if health == 0 or error_rate > 0.55:
            return 4 if info["deployment_risk"] > 0 else 3

        # If load is low and capacity is probably wasteful, scale down.
        if load_ratio < 0.70 and capacity > 1:
            return 2

        # If the current step is already expensive, trim cost when safe.
        if step_cost > 5.5 and load_ratio < 1.0 and capacity > 1:
            return 2

        # If demand is rising and we are near saturation, scale up.
        if load_ratio > 1.15 and capacity < max_capacity:
            return 1

        # Keep things stable otherwise.
        if traffic_demand < 0.40 and capacity > 2:
            return 2

        return 0
