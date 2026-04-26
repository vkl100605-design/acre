from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(".")
BASE_URL = "http://127.0.0.1:8000"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def request_json(method: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def request_text(path: str) -> str:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as response:
        return response.read().decode("utf-8")


def wait_for_server(timeout_seconds: int = 20) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            health = request_json("GET", "/health")
            if health.get("status") == "healthy":
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Server did not start in time: {last_error}")


def check_manifest() -> None:
    text = (ROOT / "openenv.yaml").read_text(encoding="utf-8")
    for field in ["name:", "version:", "runtime:", "tasks:", "endpoints:", "reset:", "step:", "state:", "health:"]:
        if field not in text:
            raise ValueError(f"openenv.yaml missing field: {field}")


def parse_score(line: str) -> float:
    match = re.search(r"score=([0-9]*\.?[0-9]+)", line)
    if not match:
        raise ValueError(f"Could not parse score from line: {line}")
    return float(match.group(1))


def check_openenv_api() -> None:
    health = request_json("GET", "/health")
    if health.get("status") != "healthy":
        raise RuntimeError("/health returned an invalid response")

    # Space bundle may serve Streamlit on public `/` via nginx; Uvicorn still exposes `/` and `/api` on :8000
    root: dict = {}
    for path in ("/", "/api"):
        try:
            root = request_json("GET", path)
            if root.get("status") == "ok":
                break
        except Exception:  # noqa: BLE001
            continue
    if root.get("status") != "ok":
        raise RuntimeError("GET / or GET /api must return JSON with status: ok for OpenEnv API")

    reset = request_json("POST", "/reset", {"task_id": "stability_spike", "seed": 11})
    if "observation" not in reset or "info" not in reset or "done" not in reset:
        raise RuntimeError("/reset response is invalid")

    state = request_json("GET", "/state")
    for field in ("task_id", "seed", "observation", "done", "info"):
        if field not in state:
            raise RuntimeError(f"/state missing field: {field}")

    if not isinstance(state.get("info"), dict):
        raise RuntimeError("/state info is not a JSON object")

    step = request_json("POST", "/step", {"action": "hold"})
    for field in ("observation", "reward", "done", "terminated", "truncated", "info", "state"):
        if field not in step:
            raise RuntimeError(f"/step missing field: {field}")

    reward = float(step.get("reward", 0.0))
    if not isinstance(reward, float):
        raise RuntimeError("/step reward is invalid")

    nested_state = step.get("state", {})
    if not isinstance(nested_state, dict) or "observation" not in nested_state:
        raise RuntimeError("/state response is invalid")

    if not isinstance(step.get("info"), dict):
        raise RuntimeError("/step info is not a JSON object")


def check_inference() -> None:
    cmd = [sys.executable, "inference.py", "--tasks", "stability_spike", "--quiet"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    end_lines = [line for line in result.stdout.splitlines() if line.startswith("[END]")]
    if not end_lines:
        raise RuntimeError("inference.py did not emit any [END] lines")
    task_lines = [line for line in end_lines if "task_id=stability_spike" in line]
    if not task_lines:
        raise RuntimeError("inference.py did not emit a task completion line")
    score = parse_score(task_lines[0])
    if not (0.0 <= score <= 1.0):
        raise RuntimeError(f"inference.py score out of range: {score}")


def main() -> None:
    for name in [
        "inference.py",
        "train.py",
        "openenv.yaml",
        "Dockerfile",
        "openenv_server.py",
        "openenv_adapter.py",
        "tasks.py",
        "env.py",
        "cost_agent.py",
        "latency_agent.py",
    ]:
        require_file(ROOT / name)

    check_manifest()

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "openenv_server:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server()
        check_openenv_api()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    check_inference()

    print("[VALIDATION] required files present")
    print("[VALIDATION] openenv.yaml includes required OpenEnv runtime + endpoint sections")
    print("[VALIDATION] openenv_server boots and responds on /, /health, /reset, /step, /state")
    print("[VALIDATION] inference.py runs and emits a normalized 0.0-1.0 score")
    print("[VALIDATION] submission looks ready for packaging")


if __name__ == "__main__":
    main()
