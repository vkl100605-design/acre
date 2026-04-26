from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Union

from cost_agent import RuleBasedCostAgent
from env import ACTION_NAMES, ACREEnv
from latency_agent import LatencyAgent
from tasks import get_task


ACTION_TO_INDEX = {name: idx for idx, name in ACTION_NAMES.items()}


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class OpenEnvAdapter:
    """
    OpenEnv-friendly wrapper around ACREEnv.

    Keeps original environment logic untouched while exposing:
    - reset()
    - step(action)
    - state()
    """

    def __init__(self, task_id: str = "incident_recovery", seed: Optional[int] = None) -> None:
        self.task = get_task(task_id)
        self.seed = int(seed) if seed is not None else self.task.seed
        self.env = ACREEnv(episode_length=self.task.episode_length, seed=self.seed)
        self.cost_agent = RuleBasedCostAgent()
        self.latency_agent = LatencyAgent()
        self.current_obs: Optional[int] = None
        self.current_info: Dict[str, Any] = {}
        self.done = False

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
        self.current_obs, self.current_info = self.env.reset(seed=self.seed)
        self.done = False
        return {
            "observation": int(self.current_obs),
            "info": _to_jsonable(self.current_info),
            "done": False,
        }

    def step(self, action: Union[int, str]) -> Dict[str, Any]:
        if self.current_obs is None:
            self.reset(seed=self.seed)

        if isinstance(action, str):
            action_idx = ACTION_TO_INDEX.get(action.strip().lower(), 0)
        else:
            action_idx = int(action)
        action_idx = int(max(0, min(self.env.action_space_n - 1, action_idx)))

        cost_action = self.cost_agent.suggest_action(self.current_info)
        latency_action = self.latency_agent.suggest_action(self.current_info)

        next_obs, reward, terminated, truncated, next_info = self.env.step(
            action_idx,
            cost_action=cost_action,
            latency_action=latency_action,
        )
        self.current_obs = int(next_obs)
        self.current_info = next_info
        self.done = bool(terminated or truncated)

        return {
            "observation": self.current_obs,
            "reward": float(reward),
            "done": self.done,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": _to_jsonable(next_info),
            "state": self.state(),
        }

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.id,
            "seed": self.seed,
            "observation": int(self.current_obs) if self.current_obs is not None else None,
            "done": bool(self.done),
            "info": _to_jsonable(self.current_info),
        }
