from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from evaluation import run_episode, save_training_artifacts, train_q_learning
import plotting
from rl_agent import load_agent, save_agent
from tasks import get_task


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "q_table.json"
TRAINING_HISTORY_PATH = ARTIFACT_DIR / "training_history_incident_recovery.json"
TRAINING_SUMMARY_PATH = ARTIFACT_DIR / "training_summary_incident_recovery.json"


def _to_history_dicts(history: List[Any]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in history]


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_training_result(result: Dict[str, Any]) -> None:
    save_training_artifacts(
        result,
        model_path=MODEL_PATH,
        history_path=TRAINING_HISTORY_PATH,
        summary_path=TRAINING_SUMMARY_PATH,
    )
    save_agent(
        MODEL_PATH,
        result["task_id"],
        result["agent"],
        {
            "trained_eval": result["trained_eval"],
            "random_eval": result["random_eval"],
            "reward_improvement_pct": result["reward_improvement_pct"],
            "cost_reduction": result["cost_reduction"],
            "uptime_improvement": result["uptime_improvement"],
            "latency_improvement_ms": result["latency_improvement_ms"],
        },
    )


def _init_state() -> None:
    if "incident_result" not in st.session_state:
        st.session_state["incident_result"] = None
    if "incident_trace" not in st.session_state:
        st.session_state["incident_trace"] = []
    if "budget_trace" not in st.session_state:
        st.session_state["budget_trace"] = []
    if "incident_agent" not in st.session_state:
        st.session_state["incident_agent"] = load_agent(MODEL_PATH)
    if "demo_mode_ran" not in st.session_state:
        st.session_state["demo_mode_ran"] = False


def _get_learning_data() -> tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    if st.session_state["incident_result"] is not None:
        result = st.session_state["incident_result"]
        return result, _to_history_dicts(result["history"])
    summary = _load_json(TRAINING_SUMMARY_PATH)
    history = _load_json(TRAINING_HISTORY_PATH)
    if isinstance(summary, dict) and isinstance(history, list):
        return summary, history
    return None, None


def _render_latency_override_banner() -> None:
    """High-emphasis banner for UX-critical latency overrides (Streamlit st.error is easy to miss)."""
    st.markdown(
        """
        <div style="
            background: linear-gradient(180deg, #dc2626 0%, #7f1d1d 100%);
            color: #fef2f2;
            border: 4px solid #fecaca;
            border-radius: 16px;
            padding: 22px 26px;
            text-align: center;
            margin: 14px 0 10px 0;
            box-shadow: 0 6px 28px rgba(220, 38, 38, 0.55);
        ">
            <div style="font-size: 1.05rem; font-weight: 800; letter-spacing: 0.12em; color: #fecaca;">
                USER EXPERIENCE PRIORITY
            </div>
            <div style="font-size: 2.05rem; font-weight: 900; line-height: 1.15; margin-top: 10px;
                text-shadow: 0 2px 12px rgba(0,0,0,0.45);">
                ⚠️ LATENCY AGENT OVERRIDE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
        <h1 style='text-align:center;'>🚀 ACRE++</h1>
        <p style='text-align:center; font-size:18px;'>
        AI learning cloud trade-offs between cost, reliability, and user experience (latency)
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "We model real-world cloud systems where AI must balance cost, reliability, and user experience. "
        "The latency agent introduces user-centric constraints, creating complex multi-agent conflicts that "
        "the RL agent must learn to handle."
    )
    st.caption("Reward = uptime - cost - latency penalty")


def _render_metrics(result: Optional[Dict[str, Any]]) -> None:
    reward_improvement_pct = float(result.get("reward_improvement_pct", 0.0)) if result else 0.0
    cost_reduction = float(result.get("cost_reduction", 0.0)) if result else 0.0
    uptime_improvement = float(result.get("uptime_improvement", 0.0)) if result else 0.0
    latency_improvement_ms = float(result.get("latency_improvement_ms", 0.0)) if result else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reward Improvement (%)", f"{reward_improvement_pct:+.1f}%")
    c2.metric("Cost Reduction", f"{cost_reduction:+.2f}")
    c3.metric("Uptime Improvement", f"{uptime_improvement:+.1f}%")
    c4.metric("Latency Improvement (ms / step)", f"{latency_improvement_ms:+.1f}")


def _render_before_after(result: Optional[Dict[str, Any]]) -> None:
    if not result:
        st.info("Train Incident Recovery to populate Random vs Trained comparison.")
        return
    trained = result.get("trained_eval", {})
    random_eval = result.get("random_eval", {})
    st.code(
        "Random:\n"
        f"Reward: {float(random_eval.get('reward', 0.0)):.2f} | Cost: {float(random_eval.get('cost', 0.0)):.2f} | "
        f"Uptime: {float(random_eval.get('uptime', 0.0)):.1f}% | "
        f"Avg latency: {float(random_eval.get('avg_latency_ms', 0.0)):.1f} ms\n\n"
        "Trained:\n"
        f"Reward: {float(trained.get('reward', 0.0)):.2f} | Cost: {float(trained.get('cost', 0.0)):.2f} | "
        f"Uptime: {float(trained.get('uptime', 0.0)):.1f}% | "
        f"Avg latency: {float(trained.get('avg_latency_ms', 0.0)):.1f} ms",
        language="text",
    )


def _render_graphs(history: Optional[List[Dict[str, Any]]]) -> None:
    st.subheader("📈 Learning Curve (Agent Improves Over Time)")
    if not history:
        st.info("Train Incident Recovery to generate graphs.")
        return
    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(plotting.plot_reward_history(history), width="stretch")
    with g2:
        st.pyplot(plotting.plot_cost_history(history), width="stretch")
    g3, g4 = st.columns(2)
    with g3:
        st.pyplot(plotting.plot_uptime_history(history), width="stretch")
    with g4:
        # Keep resilient when Streamlit hot-reload caches an older plotting module.
        latency_plot = getattr(plotting, "plot_latency_history", None)
        st.pyplot(latency_plot(history) if callable(latency_plot) else plotting.plot_cost_history(history), width="stretch")
    combo_plot = getattr(plotting, "plot_cost_latency_uptime_combo", None)
    if callable(combo_plot):
        st.pyplot(combo_plot(history), width="stretch")
    st.caption("Reward improves over time → agent learning")


def _latest_decision(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    if trace:
        return trace[-1]
    return {
        "rl_action": "-",
        "cost_action": "-",
        "latency_action": "-",
        "final_action": "-",
        "overridden": False,
        "overridden_by_latency_agent": False,
        "overridden_by_cost_agent": False,
        "latency_ms": 0.0,
        "status": "🟡",
    }


def _render_decision_box(trace: List[Dict[str, Any]]) -> None:
    st.caption("Real-time decision under conflicting objectives")
    st.markdown("### Live Decision Box")
    latest = _latest_decision(trace)
    latency_ms = float(latest.get("latency_ms", 0.0))
    override_latency = bool(latest.get("overridden_by_latency_agent", False))
    override_cost = bool(latest.get("overridden_by_cost_agent", False))
    override_text = "Yes" if (override_latency or override_cost) else "No"
    st.markdown(
        f"""
        <div style="background:#0f172a;color:#e2e8f0;border-radius:16px;padding:20px;text-align:center;">
            <div style="font-size:1.25rem;font-weight:800;margin-bottom:0.6rem;">Decision Snapshot</div>
            <div style="font-size:1.0rem;margin:0.25rem 0;"><b>RL decision:</b> {latest.get('rl_action', '-')}</div>
            <div style="font-size:1.0rem;margin:0.25rem 0;"><b>Cost decision:</b> {latest.get('cost_action', '-')}</div>
            <div style="font-size:1.0rem;margin:0.25rem 0;"><b>Latency decision:</b> {latest.get('latency_action', '-')}</div>
            <div style="font-size:1.0rem;margin:0.25rem 0;"><b>Latency:</b> {latency_ms:.0f} ms</div>
            <div style="font-size:1.1rem;color:#93c5fd;margin:0.25rem 0;"><b>Final decision:</b> {latest.get('final_action', '-')}</div>
            <div style="font-size:1.0rem;margin-top:0.35rem;"><b>Override:</b> {override_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if override_latency:
        _render_latency_override_banner()
    elif override_cost:
        st.error("⚠️ COST AGENT OVERRIDE → Budget Priority")
    else:
        st.success("✅ RL decision applied")

    st.markdown(f"### 🎯 FINAL DECISION: {str(latest.get('final_action', '-')).upper()}")


def _render_timeline(trace: List[Dict[str, Any]]) -> None:
    st.markdown("### Timeline")
    if not trace:
        st.info("Run demo to populate timeline.")
        return
    lines = []
    for item in trace[:10]:
        latency = float(item.get("latency_ms", 0.0))
        lines.append(f"Step {item['step']} → {item['status']} | {latency:.0f} ms")
    st.code("\n".join(lines), language="text")


def _render_insight() -> None:
    st.markdown("### Insight")
    st.error(
        "💡 Multi-agent conflicts now span cost, reliability, and latency: strict budgets can fight UX escalations, "
        "and RL must learn a policy that survives those collisions."
    )


def _run_incident_demo(seed: int) -> None:
    active_agent = st.session_state["incident_agent"]
    _, trace = run_episode(
        get_task("incident_recovery"),
        seed=seed,
        q_agent=active_agent,
        epsilon=0.0,
        train=False,
        random_policy=active_agent is None,
        collect_trace=True,
    )
    st.session_state["incident_trace"] = [asdict(item) for item in trace]


def _run_budget_demo(seed: int) -> None:
    active_agent = st.session_state["incident_agent"]
    _, trace = run_episode(
        get_task("budget_guardrail"),
        seed=seed,
        q_agent=active_agent,
        epsilon=0.0,
        train=False,
        random_policy=active_agent is None,
        collect_trace=True,
    )
    st.session_state["budget_trace"] = [asdict(item) for item in trace]


def main() -> None:
    st.set_page_config(page_title="ACRE++", page_icon="🚀", layout="wide")
    _init_state()
    _render_hero()

    with st.sidebar:
        with st.expander("⚙️ Controls", expanded=False):
            train_episodes = st.slider("Incident Training Episodes", min_value=80, max_value=120, value=100)
            train_seed = st.number_input("Training Seed", min_value=1, max_value=100000, value=42, step=1)
            demo_seed = st.number_input("Demo Seed", min_value=1, max_value=100000, value=99, step=1)

            if st.button("Train Incident Recovery", width="stretch", type="primary"):
                with st.spinner("Training Incident Recovery..."):
                    result = train_q_learning(
                        get_task("incident_recovery"),
                        episodes=int(train_episodes),
                        seed=int(train_seed),
                        alpha=0.1,
                        gamma=0.9,
                    )
                    st.session_state["incident_result"] = result
                    st.session_state["incident_agent"] = result["agent"]
                    _save_training_result(result)
                st.success("Incident Recovery training complete.")

            if st.button("Run Incident Demo", width="stretch"):
                _run_incident_demo(int(demo_seed))
                st.success("Incident demo complete.")

            if st.button("Run Budget Failure Demo", width="stretch"):
                _run_budget_demo(int(demo_seed) + 1)
                st.success("Budget failure demo complete.")

            st.markdown("---")
            if st.button("🎬 Run 30s Demo", width="stretch", type="primary"):
                with st.spinner("Running complete demo flow..."):
                    if st.session_state["incident_result"] is None:
                        result = train_q_learning(
                            get_task("incident_recovery"),
                            episodes=int(train_episodes),
                            seed=int(train_seed),
                            alpha=0.1,
                            gamma=0.9,
                        )
                        st.session_state["incident_result"] = result
                        st.session_state["incident_agent"] = result["agent"]
                        _save_training_result(result)
                    # Auto flow: training summary -> incident -> failure -> insight
                    _run_incident_demo(int(demo_seed))
                    _run_budget_demo(int(demo_seed) + 1)
                    st.session_state["demo_mode_ran"] = True
                st.success("30s demo flow complete.")

    result, history = _get_learning_data()
    incident_trace = st.session_state.get("incident_trace", [])
    budget_trace = st.session_state.get("budget_trace", [])

    st.markdown("### Metrics")
    _render_metrics(result)
    _render_before_after(result)
    _render_graphs(history)
    _render_decision_box(incident_trace)
    _render_timeline(incident_trace if incident_trace else budget_trace)

    if incident_trace:
        max_latency = max(float(step.get("latency_ms", 0.0)) for step in incident_trace)
        latency_hits = sum(1 for step in incident_trace if bool(step.get("overridden_by_latency_agent")))
        st.markdown("### Demo storyline (latency)")
        st.info(
            "Typical arc under Incident Recovery: normal operation → traffic/noise pushes latency up → "
            "the latency agent escalates scale-up on critical steps → cost/RL may disagree → final outcome "
            "depends on which override fired."
        )
        st.caption(
            f"Observed in this run: peak latency ≈ {max_latency:.0f} ms | latency-driven overrides: {latency_hits}"
        )

    combined_trace = incident_trace + budget_trace
    if any(bool(step.get("overridden_by_latency_agent")) for step in combined_trace):
        _render_latency_override_banner()
    elif any(bool(step.get("overridden_by_cost_agent")) for step in combined_trace):
        st.error("⚠️ COST AGENT OVERRIDE → Multi-Agent Conflict")

    if budget_trace:
        final_state = str(budget_trace[-1].get("uptime_state", "degraded"))
        if final_state == "down":
            st.error("Budget Guardrail outcome: 🔴 Failure scenario reproduced.")
        else:
            st.warning("Budget Guardrail outcome: unstable behavior observed.")

    _render_insight()


if __name__ == "__main__":
    main()
