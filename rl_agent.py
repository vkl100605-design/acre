from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Dict, List


@dataclass
class QLearningAgent:
    num_states: int
    num_actions: int
    alpha: float = 0.1
    gamma: float = 0.9
    q_table: List[List[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.q_table = [[0.0 for _ in range(self.num_actions)] for _ in range(self.num_states)]

    def act(self, state: int, rng: random.Random, epsilon: float) -> int:
        if rng.random() < epsilon:
            return rng.randrange(self.num_actions)
        return self.best_action(state)

    def best_action(self, state: int) -> int:
        values = self.q_table[state]
        best_idx = 0
        best_value = values[0]
        for idx, value in enumerate(values[1:], start=1):
            if value > best_value:
                best_value = value
                best_idx = idx
        return best_idx

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        best_next = 0.0 if done else max(self.q_table[next_state])
        target = reward + self.gamma * best_next
        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.alpha * (target - old_value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q_table": self.q_table,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "QLearningAgent":
        agent = cls(
            num_states=int(payload["num_states"]),
            num_actions=int(payload["num_actions"]),
            alpha=float(payload.get("alpha", 0.1)),
            gamma=float(payload.get("gamma", 0.9)),
        )
        agent.q_table = [[float(value) for value in row] for row in payload["q_table"]]
        return agent


def save_agent(path: Path, task_id: str, agent: QLearningAgent, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"task_id": task_id, "model": agent.to_dict(), "metadata": metadata}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_agent(path: Path) -> QLearningAgent | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model", payload)
        return QLearningAgent.from_dict(model)
    except Exception:
        return None
