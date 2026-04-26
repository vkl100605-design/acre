"""
Final submission sanity pass: artifacts, TRL dataset, OpenEnv API (in-process),
and three demo-style episodes. Prints a JSON summary to stdout.
Does not retrain models.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def _artifacts_ready() -> bool:
    need = [ART / "reward_curve.png", ART / "latency_curve.png", ART / "trl_interactions.jsonl"]
    return all(p.is_file() and p.stat().st_size > 0 for p in need)


def _regenerate_artifacts_if_needed() -> None:
    if not (ART / "reward_curve.png").is_file() or not (ART / "latency_curve.png").is_file():
        subprocess.check_call(
            [sys.executable, str(ROOT / "train_clean.py"), "--episodes", "45", "--task-id", "incident_recovery", "--seed", "42"],
            cwd=str(ROOT),
        )
    path = ART / "trl_interactions.jsonl"
    n = 0
    if path.is_file():
        n = sum(1 for _ in path.read_text(encoding="utf-8").splitlines() if _.strip())
    if n < 50:
        subprocess.check_call([sys.executable, str(ROOT / "trl_unsloth_bridge.py")], cwd=str(ROOT))


def _trl_ready() -> tuple[bool, str]:
    path = ART / "trl_interactions.jsonl"
    if not path.is_file():
        return False, "missing file"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) < 50:
        return False, f"only {len(rows)} rows (need >=50)"
    for k in ("state_text", "action", "reward"):
        if k not in rows[0]:
            return False, f"missing key {k!r} in first row"
    return True, f"{len(rows)} rows"


def _api_ready() -> tuple[bool, str]:
    try:
        from fastapi.testclient import TestClient

        from openenv_server import app

        c = TestClient(app)
        h = c.get("/health")
        if h.status_code != 200 or h.json().get("status") != "healthy":
            return False, f"/health {h.status_code} {h.text[:200]}"
        r = c.post("/reset", json={"task_id": "incident_recovery", "seed": 7})
        if r.status_code != 200 or "observation" not in r.json():
            return False, f"/reset {r.status_code}"
        s = c.get("/state")
        if s.status_code != 200 or "info" not in s.json():
            return False, f"/state {s.status_code}"
        t = c.post("/step", json={"action": "hold"})
        if t.status_code != 200 or "reward" not in t.json():
            return False, f"/step {t.status_code}"
        m = c.get("/api")
        if m.status_code != 200 or m.json().get("status") != "ok":
            return False, f"/api {m.status_code}"
        return True, "reset/step/state/health/api ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200]


def _demo_ready() -> tuple[bool, str]:
    try:
        from dataclasses import asdict

        from evaluation import run_episode
        from rl_agent import load_agent
        from tasks import get_task

        agent = load_agent(ART / "q_table_clean.json")
        if agent is None:
            agent = load_agent(ART / "q_table.json")

        # Balanced / typical: incident_recovery
        _, tr1 = run_episode(
            get_task("incident_recovery"),
            seed=42,
            q_agent=agent,
            epsilon=0.0,
            train=False,
            random_policy=False,
            collect_trace=True,
        )
        if not tr1:
            return False, "incident_recovery empty trace"

        # Traffic / latency pressure: stability_spike
        _, tr2 = run_episode(
            get_task("stability_spike"),
            seed=11,
            q_agent=agent,
            epsilon=0.0,
            train=False,
            random_policy=False,
            collect_trace=True,
        )
        if not tr2:
            return False, "stability_spike empty trace"

        # Cost / budget stress: budget_guardrail
        _, tr3 = run_episode(
            get_task("budget_guardrail"),
            seed=99,
            q_agent=agent,
            epsilon=0.0,
            train=False,
            random_policy=False,
            collect_trace=True,
        )
        if not tr3:
            return False, "budget_guardrail empty trace"

        def _flags(trace):
            rows = [asdict(x) for x in trace]
            lat = any(bool(x.get("overridden_by_latency_agent")) for x in rows)
            cost = any(bool(x.get("overridden_by_cost_agent")) for x in rows)
            return lat, cost

        lat1, c1 = _flags(tr1)
        lat2, c2 = _flags(tr2)
        lat3, c3 = _flags(tr3)
        # At least one scenario should show an override somewhere (soft check)
        if not (lat1 or c1 or lat2 or c2 or lat3 or c3):
            return True, "3 episodes ok (no overrides this seed — acceptable)"
        return True, "3 episodes ok with visible override metadata in at least one run"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:300]


def main() -> None:
    _regenerate_artifacts_if_needed()
    env_ready = (ROOT / "env.py").is_file() and (ROOT / "openenv_server.py").is_file()
    art_ok = _artifacts_ready()
    trl_ok, trl_msg = _trl_ready()
    api_ok, api_msg = _api_ready()
    demo_ok, demo_msg = _demo_ready()

    summary = {
        "env_ready": bool(env_ready),
        "api_ready": bool(api_ok),
        "artifacts_ready": bool(art_ok),
        "trl_ready": bool(trl_ok),
        "demo_ready": bool(demo_ok),
    }
    print(json.dumps(summary, indent=2))
    if not all(summary.values()):
        print(
            json.dumps({"_debug": {"trl": trl_msg, "api": api_msg, "demo": demo_msg}}, indent=2),
            file=sys.stderr,
        )
    if not all([env_ready, api_ok, art_ok, trl_ok, demo_ok]):
        sys.exit(1)


if __name__ == "__main__":
    main()
