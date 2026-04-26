from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Optional, Tuple


ACTION_NAMES = {
    0: "hold",
    1: "scale_up",
    2: "scale_down",
    3: "restart",
    4: "rollback",
}


@dataclass
class StepInfo:
    traffic_demand: float
    traffic_bucket: int
    capacity: int
    health: int
    deployment_risk: int
    load_ratio: float
    error_rate: float
    step_cost: float
    final_action: int
    rl_action: int
    cost_action: Optional[int]
    overridden_by_cost_agent: bool
    latency_ms: float
    latency_action: Optional[int]
    overridden_by_latency_agent: bool


class ACREEnv:
    """
    Minimal cloud reliability vs cost simulator.

    Hidden state:
    - traffic demand, capacity, health, deployment risk

    Observation returned to the agent:
    - encoded discrete tuple of noisy traffic, error, cost, risk, health buckets
    """

    def __init__(
        self,
        episode_length: int = 60,
        min_capacity: int = 1,
        max_capacity: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        self.episode_length = episode_length
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity

        self.action_space_n = 5
        self.state_sizes = (4, 4, 4, 3, 3)  # traffic, error, cost, risk, health
        self.observation_space_n = 1
        for size in self.state_sizes:
            self.observation_space_n *= size

        self.rng = random.Random(seed)

        self.t = 0
        self.capacity = 3
        self.health = 2  # 0=down, 1=degraded, 2=healthy
        self.deployment_risk = 0  # 0=low, 1=medium, 2=high
        self.pending_capacity_delta = 0
        self.traffic_demand = 0.5
        self.last_error_rate = 0.0
        self.last_step_cost = 0.0
        self.last_latency_ms = 0.0
        self.last_action = 0
        self.last_cost_action: Optional[int] = None
        self.last_latency_action: Optional[int] = None
        self.done = False
        self.traffic_phase = 0.0

    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict]:
        if seed is not None:
            self.rng = random.Random(seed)

        self.t = 0
        self.capacity = 3
        self.health = 2
        self.deployment_risk = 0
        self.pending_capacity_delta = 0
        self.traffic_demand = float(self.rng.uniform(0.35, 0.75))
        self.last_error_rate = 0.0
        self.last_step_cost = 0.0
        self.last_latency_ms = 0.0
        self.last_action = 0
        self.last_cost_action = None
        self.last_latency_action = None
        self.done = False
        self.traffic_phase = float(self.rng.uniform(0.0, 2.0 * math.pi))

        obs = self._get_observation()
        return obs, self._build_info(
            final_action=0,
            rl_action=0,
            cost_action=None,
            latency_action=None,
            overridden=False,
            override_reason="none",
            override_agent="none",
            conflict_type="none",
        )

    def step(
        self,
        action: int,
        cost_action: Optional[int] = None,
        latency_action: Optional[int] = None,
    ) -> Tuple[int, float, bool, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        rl_action = int(action)
        final_action, override_meta = self._resolve_multi_agent(rl_action, cost_action, latency_action)
        overridden = bool(override_meta.get("overridden", False))
        override_reason = str(override_meta.get("override_reason", "none"))
        override_agent = str(override_meta.get("override_agent", "none"))
        conflict_type = str(override_meta.get("conflict_type", "none"))

        self.t += 1
        previous_health = self.health

        # Apply delayed scaling from the previous step.
        if self.pending_capacity_delta != 0:
            self.capacity = int(self._clip(self.capacity + self.pending_capacity_delta, self.min_capacity, self.max_capacity))
            self.pending_capacity_delta = 0

        # Traffic is partly predictable but noisy.
        self.traffic_demand = self._next_traffic_demand()
        demand_units = 1.0 + 4.0 * self.traffic_demand
        load_ratio = demand_units / float(self.capacity)

        # Failure injection: overload and risky deployment behavior.
        bug_prob = 0.03 + 0.06 * self.deployment_risk
        bug_prob += max(0.0, load_ratio - 1.0) * 0.18
        if final_action in (1, 3, 4):
            bug_prob += 0.05
        if self.rng.random() < min(0.75, bug_prob):
            self.deployment_risk = min(2, self.deployment_risk + 1)
            self.health = max(0, self.health - 1)

        # Action effects, with delayed consequences for scaling.
        action_cost = 0.0
        if final_action == 1:  # scale_up
            self.pending_capacity_delta += 1
            action_cost += 0.6
        elif final_action == 2:  # scale_down
            self.pending_capacity_delta -= 1
            action_cost += 0.35
        elif final_action == 3:  # restart
            action_cost += 0.9
            if self.health < 2 and self.rng.random() < 0.72:
                self.health = min(2, self.health + 1)
                self.deployment_risk = max(0, self.deployment_risk - 1)
        elif final_action == 4:  # rollback
            action_cost += 1.1
            self.health = min(2, max(self.health, 1))
            self.deployment_risk = max(0, self.deployment_risk - 1)

        # Overload can still harm the service even after the action is applied.
        overload_severity = max(0.0, load_ratio - 1.0)
        if overload_severity > 0.0:
            degrade_prob = min(0.85, overload_severity * 0.55)
            if self.rng.random() < degrade_prob:
                self.health = max(0, self.health - 1)

        if load_ratio > 1.65 and self.rng.random() < 0.35:
            self.health = 0

        # Mild autonomous recovery when load is under control.
        if self.health == 0 and load_ratio < 0.95 and self.rng.random() < 0.18:
            self.health = 1
        elif self.health == 1 and load_ratio < 0.85 and self.rng.random() < 0.25:
            self.health = 2

        # Cost model: infrastructure cost grows with capacity and high demand.
        price_multiplier = 1.0 + 0.25 * self.traffic_demand
        infra_cost = float(self.capacity) * price_multiplier
        incident_cost = 2.0 if self.health == 0 else 0.8 if self.health == 1 else 0.0
        self.last_step_cost = infra_cost + incident_cost + action_cost

        # Error rate is partially observed and noisy.
        base_error = 0.03 if self.health == 2 else 0.18 if self.health == 1 else 0.55
        error_noise = float(self.rng.gauss(0.0, 0.03))
        self.last_error_rate = float(self._clip(base_error + 0.22 * overload_severity + error_noise, 0.0, 1.0))

        self.last_latency_ms = self._compute_latency_ms(load_ratio=load_ratio)

        reward = self._compute_reward(load_ratio, previous_health)

        self.last_action = final_action
        self.last_cost_action = cost_action
        self.last_latency_action = latency_action
        terminated = self.t >= self.episode_length
        truncated = False
        self.done = terminated or truncated

        obs = self._get_observation()
        info = self._build_info(
            final_action=final_action,
            rl_action=rl_action,
            cost_action=cost_action,
            latency_action=latency_action,
            overridden=overridden,
            override_reason=override_reason,
            override_agent=override_agent,
            conflict_type=conflict_type,
            load_ratio=load_ratio,
        )
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, load_ratio: float, previous_health: int) -> float:
        # Reward balances uptime, cost efficiency, overload penalties, and recovery behavior.
        uptime_score = 2.0 if self.health == 2 else 0.6 if self.health == 1 else -2.8
        overload_penalty = 0.9 * max(0.0, load_ratio - 1.0)
        waste_penalty = 0.15 * max(0.0, 1.0 - load_ratio)
        cost_penalty = 0.12 * self.last_step_cost
        recovery_bonus = 0.8 if previous_health == 0 and self.health > 0 else 0.0
        error_penalty = 1.0 * self.last_error_rate
        latency_penalty = 0.0035 * float(self.last_latency_ms)

        return uptime_score + recovery_bonus - overload_penalty - waste_penalty - cost_penalty - error_penalty - latency_penalty

    def _compute_latency_ms(self, *, load_ratio: float) -> float:
        # Latency rises with load and errors, and improves when capacity is higher.
        base = 55.0
        load_component = 220.0 * max(0.0, load_ratio - 0.85) ** 1.35
        error_component = 260.0 * float(self.last_error_rate)
        health_component = 0.0 if self.health == 2 else 55.0 if self.health == 1 else 140.0
        capacity_relief = 18.0 * float(self.capacity - self.min_capacity)
        noise = float(self.rng.gauss(0.0, 6.0))
        latency = base + load_component + error_component + health_component - capacity_relief + noise
        return float(self._clip(latency, 25.0, 900.0))

    def _next_traffic_demand(self) -> float:
        # Diurnal pattern + randomness + occasional spike.
        phase = (2.0 * math.pi * (self.t + 1) / float(self.episode_length)) + self.traffic_phase
        wave = 0.5 + 0.28 * math.sin(phase)
        noise = float(self.rng.gauss(0.0, 0.06))
        spike = 0.0
        if self.rng.random() < 0.08:
            spike = float(self.rng.uniform(0.18, 0.35))
        return float(self._clip(wave + noise + spike, 0.0, 1.0))

    def _latency_is_critical(self) -> bool:
        demand_units = 1.0 + 4.0 * self.traffic_demand
        projected_load = demand_units / float(max(self.capacity + 1, self.min_capacity))
        return bool(self.last_latency_ms >= 220.0 or projected_load > 1.05)

    def _classify_conflict(
        self,
        *,
        rl_action: int,
        cost_action: Optional[int],
        latency_action: Optional[int],
        final_action: int,
        override_reason: str,
    ) -> str:
        if override_reason == "none":
            return "none"

        proposals = {int(rl_action)}
        if cost_action is not None:
            proposals.add(int(cost_action))
        if latency_action is not None:
            proposals.add(int(latency_action))

        if len(proposals) <= 1:
            return "none"

        if override_reason == "latency_override":
            if cost_action is not None and int(cost_action) != int(rl_action) and int(cost_action) != int(final_action):
                return "latency_vs_cost"
            return "latency_vs_rl"

        if override_reason == "cost_override":
            if latency_action is not None and int(latency_action) != int(rl_action) and int(latency_action) != int(final_action):
                return "cost_vs_latency"
            return "cost_vs_rl"

        return "multi_disagreement"

    def _resolve_multi_agent(
        self,
        rl_action: int,
        cost_action: Optional[int],
        latency_action: Optional[int],
    ) -> Tuple[int, Dict[str, object]]:
        # Latency-first override when UX is critical and RL isn't already scaling up.
        if latency_action is not None and int(latency_action) == 1 and int(rl_action) != 1 and self._latency_is_critical():
            final_action = 1
            meta = {
                "overridden": True,
                "override_reason": "latency_override",
                "override_agent": "latency",
                "conflict_type": self._classify_conflict(
                    rl_action=rl_action,
                    cost_action=cost_action,
                    latency_action=latency_action,
                    final_action=final_action,
                    override_reason="latency_override",
                ),
            }
            return final_action, meta

        if cost_action is None:
            meta = {
                "overridden": False,
                "override_reason": "none",
                "override_agent": "none",
                "conflict_type": "none",
            }
            return int(rl_action), meta

        # Safety-first override when the service is already unhealthy.
        if self.health == 0 and cost_action in (3, 4):
            final_action = int(cost_action)
            meta = {
                "overridden": final_action != int(rl_action),
                "override_reason": "cost_override",
                "override_agent": "cost",
                "conflict_type": self._classify_conflict(
                    rl_action=rl_action,
                    cost_action=cost_action,
                    latency_action=latency_action,
                    final_action=final_action,
                    override_reason="cost_override",
                ),
            }
            return final_action, meta

        # Cost agent can override wasteful scaling when demand is low.
        if cost_action == 2 and rl_action == 1 and self.traffic_demand < 0.62:
            final_action = 2
            meta = {
                "overridden": True,
                "override_reason": "cost_override",
                "override_agent": "cost",
                "conflict_type": self._classify_conflict(
                    rl_action=rl_action,
                    cost_action=cost_action,
                    latency_action=latency_action,
                    final_action=final_action,
                    override_reason="cost_override",
                ),
            }
            return final_action, meta

        # Reliability-first override when the service is in trouble.
        if cost_action in (3, 4) and self.health < 2:
            final_action = int(cost_action)
            meta = {
                "overridden": final_action != int(rl_action),
                "override_reason": "cost_override",
                "override_agent": "cost",
                "conflict_type": self._classify_conflict(
                    rl_action=rl_action,
                    cost_action=cost_action,
                    latency_action=latency_action,
                    final_action=final_action,
                    override_reason="cost_override",
                ),
            }
            return final_action, meta

        # If the cost agent wants to scale up while RL wants to scale down,
        # prefer the side that better matches current demand.
        if cost_action == 1 and rl_action == 2:
            final_action = 1 if self.traffic_demand > 0.68 else 2
            meta = {
                "overridden": final_action != int(rl_action),
                "override_reason": "cost_override",
                "override_agent": "cost",
                "conflict_type": self._classify_conflict(
                    rl_action=rl_action,
                    cost_action=cost_action,
                    latency_action=latency_action,
                    final_action=final_action,
                    override_reason="cost_override",
                ),
            }
            return final_action, meta

        if cost_action == 2 and rl_action == 0 and self.capacity > self.min_capacity:
            final_action = 2
            meta = {
                "overridden": True,
                "override_reason": "cost_override",
                "override_agent": "cost",
                "conflict_type": self._classify_conflict(
                    rl_action=rl_action,
                    cost_action=cost_action,
                    latency_action=latency_action,
                    final_action=final_action,
                    override_reason="cost_override",
                ),
            }
            return final_action, meta

        meta = {
            "overridden": False,
            "override_reason": "none",
            "override_agent": "none",
            "conflict_type": self._classify_conflict(
                rl_action=rl_action,
                cost_action=cost_action,
                latency_action=latency_action,
                final_action=int(rl_action),
                override_reason="none",
            ),
        }
        return int(rl_action), meta

    def _get_observation(self) -> int:
        traffic_bucket = self._bucket(self.traffic_demand + float(self.rng.gauss(0.0, 0.05)), 4)

        error_bucket = self._bucket(
            self.last_error_rate + float(self.rng.gauss(0.0, 0.02)),
            4,
        )

        # Normalize cost into a rough 0..1 range, then bucket it.
        cost_norm = self._clip(self.last_step_cost / 8.0 + float(self.rng.gauss(0.0, 0.04)), 0.0, 1.0)
        cost_bucket = self._bucket(float(cost_norm), 4)

        risk_bucket = int(self._clip(self.deployment_risk + self.rng.randint(-1, 1), 0, 2))
        if self.rng.random() < 0.15:
            health_noisy = self.health + self.rng.randint(-1, 1)
        else:
            health_noisy = self.health
        health_bucket = int(self._clip(health_noisy, 0, 2))

        state_tuple = (
            traffic_bucket,
            error_bucket,
            cost_bucket,
            risk_bucket,
            health_bucket,
        )
        return self.encode_state(state_tuple)

    def _build_info(
        self,
        *,
        final_action: int,
        rl_action: int,
        cost_action: Optional[int],
        latency_action: Optional[int],
        overridden: bool,
        override_reason: str,
        override_agent: str,
        conflict_type: str,
        load_ratio: Optional[float] = None,
    ) -> Dict:
        demand_units = 1.0 + 4.0 * self.traffic_demand
        if load_ratio is None:
            load_ratio = demand_units / float(self.capacity)

        return {
            "step": self.t,
            "traffic_demand": self.traffic_demand,
            "traffic_bucket": self._bucket(self.traffic_demand, 4),
            "demand_units": demand_units,
            "capacity": self.capacity,
            "max_capacity": self.max_capacity,
            "health": self.health,
            "deployment_risk": self.deployment_risk,
            "load_ratio": load_ratio,
            "error_rate": self.last_error_rate,
            "step_cost": self.last_step_cost,
            "latency_ms": self.last_latency_ms,
            "final_action": final_action,
            "rl_action": rl_action,
            "cost_action": cost_action,
            "latency_action": latency_action,
            "overridden_by_cost_agent": bool(overridden and override_agent == "cost"),
            "overridden_by_latency_agent": bool(overridden and override_agent == "latency"),
            "override_reason": override_reason,
            "override_agent": override_agent,
            "conflict_type": conflict_type,
            "action_name": ACTION_NAMES.get(final_action, "unknown"),
        }

    def _bucket(self, value: float, bins: int) -> int:
        value = float(self._clip(value, 0.0, 1.0))
        return int(min(bins - 1, value * bins))

    def _clip(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def encode_state(self, state_tuple: Tuple[int, int, int, int, int]) -> int:
        idx = 0
        multiplier = 1
        for value, base in zip(state_tuple, self.state_sizes):
            idx += int(value) * multiplier
            multiplier *= base
        return idx

    def decode_state(self, index: int) -> Tuple[int, int, int, int, int]:
        values = []
        remaining = int(index)
        for base in self.state_sizes:
            values.append(remaining % base)
            remaining //= base
        return tuple(values)

    def render(self) -> None:
        print(
            f"t={self.t:02d} traffic={self.traffic_demand:.2f} capacity={self.capacity} "
            f"health={self.health} risk={self.deployment_risk} cost={self.last_step_cost:.2f}"
        )
