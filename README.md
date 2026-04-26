# ACRE++: Multi-Agent RL for Cloud Cost, Reliability, and User Experience

ACRE++ is a production-inspired simulation where an RL controller must keep cloud systems healthy while balancing:

- **Reliability** (uptime, incident recovery)
- **Cost** (FinOps pressure, scaling budget)
- **User Experience** (latency-sensitive behavior)

This repository keeps the original Streamlit demo and environment logic, and now adds OpenEnv-style compatibility and hackathon-ready training evidence.

---

## 1) Problem Statement

Modern DevOps systems are never single-objective:

- If you aggressively cut cost, uptime can collapse.
- If you scale for uptime only, spend can spike.
- If you ignore latency, users churn even when systems are technically "up."

ACRE++ models this exact tension as a multi-agent RL environment with conflicting priorities.

---

## 2) Why This Matters (DevOps + Cloud RL)

Cloud reliability incidents rarely have immediate effects. There are delayed scaling impacts, noisy telemetry, and competing operators. ACRE++ captures those realities so an RL policy can learn practical trade-offs, not toy rules.

---

## 3) Environment Design

Core simulator lives in `env.py` and includes delayed effects, failure injections, noisy observations, and dynamic latency.

### State (observed/derived)

- traffic demand and bucketed traffic
- error rate
- step cost
- capacity
- deployment risk
- health state
- latency in milliseconds (`latency_ms`)

### Action Space

- `hold`
- `scale_up`
- `scale_down`
- `restart`
- `rollback`

### Reward

ACRE++ reward is shaped to penalize expensive, unstable, and high-latency behavior:

`reward = uptime/recovery terms - overload - waste - cost - error - latency_penalty`

In UI terms:

`Reward = uptime - cost - latency penalty`

---

## 4) Multi-Agent System

ACRE++ runs three agents with explicit conflict handling:

1. **RL Agent** (`rl_agent.py`)  
   Learns long-term action values with tabular Q-learning.

2. **Cost Agent** (`cost_agent.py`)  
   Rule-based FinOps logic to reduce waste and enforce safer spend behavior.

3. **Latency Agent** (`latency_agent.py`)  
   UX-focused guardrail; escalates `scale_up` when latency crosses threshold.

When conflicts happen, override metadata is tracked (who overrode and why), enabling explainable multi-agent outcomes.

---

## 5) OpenEnv Compatibility Layer

### Adapter

- `openenv_adapter.py`
- Wraps the existing environment without rewriting internals.
- Exposes:
  - `reset()`
  - `step(action)`
  - `state()`

### API Server (FastAPI)

- `openenv_server.py`
- Endpoints:
  - `POST /reset`
  - `POST /step`
  - `GET /state`
  - plus `GET /` and `GET /health`

Run locally:

```bash
uvicorn openenv_server:app --host 0.0.0.0 --port 8000
```

---

## 6) Training Pipelines

### Existing pipeline (kept intact)

- `train.py` + `evaluation.py`
- Baseline comparison (Random vs Trained) and Streamlit visual workflow

### Clean compliance loop (added)

- `train_clean.py`
- Explicit environment interaction loop:

```python
state, info = env.reset()
while not done:
    action = agent.act(state, rng, epsilon)
    next_state, reward, terminated, truncated, info = env.step(action, ...)
```

Logs and saves:

- reward per episode
- latency per episode (avg ms per step)
- cost per episode

Artifacts in `/artifacts`:

- `reward_curve.png`
- `latency_curve.png`
- `training_clean_summary.json`

Run:

```bash
python train_clean.py --task-id incident_recovery --episodes 100 --seed 42
```

---

## 7) TRL / Unsloth Compatibility (Minimal Valid Layer)

- `trl_unsloth_bridge.py`
- Simulates rollouts and exports a dataset in `(state -> action -> reward)` format.
- Writes:
  - `artifacts/trl_interactions.jsonl`
  - `artifacts/trl_compat_report.json`

It tries to initialize a minimal TRL PPO config if `trl` is installed, otherwise produces a compatibility stub report (hackathon-friendly, non-breaking).

Run:

```bash
python trl_unsloth_bridge.py
```

---

## 8) Streamlit Demo (Kept, Not Removed)

Main UI:

```bash
streamlit run app.py
```

Demo highlights:

- **Balanced case**: RL policy aligns cost + reliability.
- **Latency override case**: UX-critical latency triggers latency-agent override.
- **Cost failure case**: budget-guardrail scenario demonstrates destabilization risk.

---

## 9) Training Results and Evidence

Precomputed artifacts are stored in `/artifacts` and can be embedded in reports/submissions:

- `reward_curve.png` — episode total reward (regenerate with `train_clean.py`)
- `latency_curve.png` — average latency per episode
- `training_clean_summary.json` — trained vs random metrics
- After an LLM+PPO run (`train_llm.py`), also:
  - `artifacts/llm_ppo/llm_env_reward_curve.png`, `llm_shaped_reward_curve.png`, `llm_loss_curve.png` (loss if logged)
- `python plot_submission_artifacts.py` — rebuilds LLM plots from `llm_training_log.jsonl` if you only have the JSONL

These provide direct evidence of learning behavior and trade-offs.

---

## 9b) Judges: what we look for & how to re-run

- **Judging notes (official)**: [What judges look for](https://docs.google.com/document/d/1Odznuzwtb1ecDOm2t6ToZd4MuMXXfO6vWUGcxbC6mFs/edit?tab=t.0#bookmark=kix.2dz0x0nie3me)

- **Colab notebook** (upload/clone this repo, then run top-to-bottom): [`ACRExx_Training_Colab.ipynb`](./ACRExx_Training_Colab.ipynb)  
  After you push to GitHub, “Open in Colab” URL pattern:  
  `https://colab.research.google.com/github/<USER>/<REPO>/blob/main/ACRExx_Training_Colab.ipynb`

- **Local training (same as Colab cells)**:
  - Q-learning + reward/latency PNGs: `python train_clean.py --episodes 80 --task-id incident_recovery --seed 42`
  - Optional LLM+PPO: `python train_llm.py --episodes 4 --task-id incident_recovery --model-name Qwen/Qwen2.5-0.5B-Instruct --seed 42`

---

## 10) Docker / Entry Point

`Dockerfile` is configured for OpenEnv API deployment and starts the FastAPI runtime.

Build and run:

```bash
docker build -t acrepp .
docker run -p 8000:8000 acrepp
```

---

## 11) Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

### Entry Points

#### Streamlit demo (visual storytelling)

```bash
streamlit run app.py
```

#### OpenEnv API (hackathon submission runtime)

```bash
uvicorn openenv_server:app --host 0.0.0.0 --port 8000
```

### Optional jobs

```bash
python train_clean.py --episodes 100
python trl_unsloth_bridge.py
python plot_submission_artifacts.py   # LLM log → PNGs (after train_llm.py)
```

---

## 12) LLM + PPO (TRL) — SRE policy (optional, GPU-recommended)

After installing heavy deps in `requirements.txt` (`trl`, `torch`, `transformers`):

**Train** (small Qwen 0.5B by default; use GPU for anything larger):

```bash
python train_llm.py --episodes 4 --task-id incident_recovery --model-name Qwen/Qwen2.5-0.5B-Instruct --seed 42
```

**Outputs** (default `artifacts/llm_ppo/`):

- `model/` — PPO value-head LLM + tokenizer
- `llm_training_log.jsonl` — step rewards + shaped reward
- `reasoning_samples.txt` — model JSON and rationale snippets
- `training_meta.json` — run config

`evaluation.run_episode(..., llm_agent=SRELLMAgent.from_checkpoint_dir(...))` uses the same environment and multi-agent cost/latency suggestions as Q-learning, but the primary action is chosen from LLM text output parsed as JSON.
