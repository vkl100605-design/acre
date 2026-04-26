from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionName(str, Enum):
    HOLD = "hold"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    RESTART = "restart"
    ROLLBACK = "rollback"


@dataclass
class ResetRequest:
    task_id: Optional[str] = None
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ResetRequest":
        data = data or {}
        return cls(task_id=data.get("task_id"), seed=data.get("seed"))


@dataclass
class StepRequest:
    action: ActionName = ActionName.HOLD
    note: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "StepRequest":
        data = data or {}
        action = data.get("action", ActionName.HOLD)
        if not isinstance(action, ActionName):
            action = ActionName(str(action))
        return cls(action=action, note=data.get("note"))


@dataclass
class TaskModel:
    id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    reward_range: List[float]
    episode_length: int
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservationModel:
    task_id: str
    step: int
    traffic_demand: float
    traffic_level: str
    capacity: int
    health: str
    deployment_risk: str
    load_ratio: float
    error_rate: float
    step_cost: float
    last_action: str
    reward: float
    score: float
    done: bool
    overridden_by_cost_agent: bool
    info: Dict[str, Any]


@dataclass
class StateModel:
    episode_id: int
    task_id: str
    step: int
    done: bool
    cumulative_reward: float
    cumulative_cost: float
    observation: ObservationModel


@dataclass
class StepResultModel:
    observation: ObservationModel
    state: StateModel
    reward: float
    score: float
    done: bool
    truncated: bool = False
    info: Dict[str, Any] = None  # type: ignore[assignment]


@dataclass
class ResetResultModel:
    observation: ObservationModel
    state: StateModel
    task: TaskModel


@dataclass
class HealthModel:
    status: str
    message: str
