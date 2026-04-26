from __future__ import annotations

from typing import Any, Dict, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel

from openenv_adapter import OpenEnvAdapter


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Union[int, str] = "hold"


app = FastAPI(title="ACRE++ OpenEnv API", version="1.0.0")
runtime = OpenEnvAdapter(task_id="incident_recovery")


@app.get("/")
def root() -> Dict[str, str]:
    """Served on direct Uvicorn; on HF Space nginx may route `/` to Streamlit instead."""
    return {"name": "ACRE++ OpenEnv API", "status": "ok"}


@app.get("/api")
def api_meta() -> Dict[str, str]:
    """Same JSON as root; use when `/` is the Streamlit UI (reverse proxy)."""
    return {"name": "ACRE++ OpenEnv API", "status": "ok"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset")
def reset(payload: ResetRequest) -> Dict[str, Any]:
    global runtime
    task_id = payload.task_id or "incident_recovery"
    runtime = OpenEnvAdapter(task_id=task_id, seed=payload.seed)
    return runtime.reset(seed=payload.seed)


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, Any]:
    return runtime.step(payload.action)


@app.get("/state")
def state() -> Dict[str, Any]:
    return runtime.state()
