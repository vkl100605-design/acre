from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TaskDefinition:
    id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    reward_range: List[float]
    episode_length: int
    seed: int

    def to_dict(self) -> Dict:
        return asdict(self)


TASKS: List[TaskDefinition] = [
    TaskDefinition(
        id="stability_spike",
        name="Stability Under Traffic Spike",
        description="Handle a sudden traffic spike while balancing reliability and cost.",
        difficulty="easy",
        max_steps=20,
        reward_range=[0.0, 1.0],
        episode_length=20,
        seed=11,
    ),
    TaskDefinition(
        id="incident_recovery",
        name="Incident Recovery",
        description="Recover a degraded service with delayed effects and noisy metrics.",
        difficulty="medium",
        max_steps=22,
        reward_range=[0.0, 1.0],
        episode_length=22,
        seed=29,
    ),
    TaskDefinition(
        id="budget_guardrail",
        name="Budget Guardrail",
        description="Minimize wasteful spending while preserving service uptime.",
        difficulty="hard",
        max_steps=24,
        reward_range=[0.0, 1.0],
        episode_length=24,
        seed=47,
    ),
]


TASKS_BY_ID: Dict[str, TaskDefinition] = {task.id: task for task in TASKS}


def get_task(task_id: Optional[str]) -> TaskDefinition:
    if task_id and task_id in TASKS_BY_ID:
        return TASKS_BY_ID[task_id]
    return TASKS[0]
