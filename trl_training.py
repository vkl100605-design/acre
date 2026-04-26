from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Avoid noisy import warning (TRL 1.x value head lives in experimental; we handle the API in code).
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
import torch.nn.functional as F

from cost_agent import RuleBasedCostAgent
from env import ACTION_NAMES, ACREEnv
from evaluation import evaluate_llm, evaluate_random
from latency_agent import LatencyAgent
from llm_agent import SRELLMAgent, build_sre_messages
from parser import parse_action_json
from reward import combined_step_reward
from tasks import get_task

# TRL: value-head was moved; classic PPOTrainer.step() was removed in TRL 1.x — we import
# AutoModelForCausalLMWithValueHead and run a small compatible PPO update loop locally.
_TRL_IMPORT_ERR: Optional[Exception] = None
AutoModelForCausalLMWithValueHead: Any = None
try:  # pragma: no cover
    from trl import AutoModelForCausalLMWithValueHead as _VHead  # type: ignore[assignment]

    AutoModelForCausalLMWithValueHead = _VHead
except Exception:  # noqa: BLE001 — ImportError and lazy-module issues
    try:
        from trl.experimental.ppo import AutoModelForCausalLMWithValueHead as _VHead  # type: ignore[assignment]

        AutoModelForCausalLMWithValueHead = _VHead
    except Exception as e:  # noqa: BLE001
        _TRL_IMPORT_ERR = e
        AutoModelForCausalLMWithValueHead = None

from transformers import AutoTokenizer


def _forward_lm_value(
    model: Any,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(input_ids=full_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
    lm_batch: torch.Tensor = out[0]
    value_batch: torch.Tensor = out[2]
    return lm_batch[0], value_batch[0]  # (L, V), (L,)

def _response_token_logprobs(
    lm_logits: torch.Tensor,
    full_1d: torch.Tensor,
    q_len: int,
) -> torch.Tensor:
    """
    For causal LM, logits at position t predict token t+1. One logprob per response token
    (same convention as early TRL PPOTrainer on query + response).
    """
    l_total = int(full_1d.shape[0])
    if l_total <= 1 or q_len >= l_total:
        return torch.zeros(0, device=full_1d.device, dtype=torch.float32)
    logits_slice = lm_logits[q_len - 1 : l_total - 1, :]
    targets = full_1d[q_len : l_total]
    log_probs = F.log_softmax(logits_slice.float(), dim=-1)
    return log_probs[range(int(targets.shape[0])), targets].clone()


def _ppo_value_head_ministep(
    model: Any,
    query: torch.Tensor,
    response: torch.Tensor,
    reward: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    cliprange: float = 0.2,
    vf_coef: float = 0.5,
) -> Dict[str, float]:
    """
    Single-trajectory PPO-style update (matches TRL 0.11: separate query/response, full LM + value).
    """
    if query.dim() != 1 or response.dim() != 1:
        return {}
    if response.numel() == 0:
        return {}
    model.train()
    full = torch.cat([query, response], dim=0)
    l_total = int(full.shape[0])
    q_len = int(query.shape[0])
    attention = torch.ones(l_total, device=device, dtype=torch.long)
    with torch.no_grad():
        old_lm, value_row = _forward_lm_value(model, full, attention)
        old_lps = _response_token_logprobs(old_lm, full, q_len)
        v_old = value_row[q_len - 1].float()
    r = float(reward)
    with torch.set_grad_enabled(True):
        new_lm, value_row2 = _forward_lm_value(model, full, attention)
        new_lps = _response_token_logprobs(new_lm, full, q_len)
    if old_lps.numel() == 0 or new_lps.numel() == 0:
        return {}
    if old_lps.shape != new_lps.shape:
        return {}
    ratio = (new_lps - old_lps).exp()
    adv = (torch.tensor(r, device=device, dtype=old_lps.dtype) - v_old).expand_as(ratio)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * adv
    policy_loss = -torch.min(surr1, surr2).mean()
    v_new = value_row2[q_len - 1].float()
    v_target = torch.tensor(r, device=device, dtype=v_new.dtype)
    value_loss = 0.5 * (v_new - v_target) ** 2
    loss = policy_loss + vf_coef * value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss/total": float(loss.item()),
        "loss/policy": float(policy_loss.item()),
        "loss/value": float(value_loss.item()),
    }


def _action_name_to_idx(name: str) -> int:
    for idx, aname in ACTION_NAMES.items():
        if aname == name:
            return int(idx)
    return 0


def _build_prompt_text(tokenizer, info: dict, cost: int, lat: Optional[int]) -> str:
    messages = build_sre_messages(
        info,
        cost_suggestion=cost,
        latency_suggestion=lat,
    )
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def _encode_query(tokenizer, prompt: str, device: torch.device, max_len: int = 1024) -> torch.Tensor:
    t = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_len,
    )["input_ids"]
    return t.squeeze(0).to(device)


def run_ppo_training(
    task_id: str,
    episodes: int,
    model_name: str,
    seed: int,
    output_dir: Path,
    *,
    max_steps_per_episode: int = 0,
) -> Dict[str, Any]:
    if _TRL_IMPORT_ERR is not None or AutoModelForCausalLMWithValueHead is None:
        raise RuntimeError(
            "TRL with AutoModelForCausalLMWithValueHead is required for PPO. "
            "Install: pip install 'trl' torch transformers datasets accelerate. "
            "On TRL 1.x, the value head lives under trl.experimental.ppo (import is handled in code)."
        ) from _TRL_IMPORT_ERR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "llm_training_log.jsonl"
    sample_path = output_dir / "reasoning_samples.txt"
    log_path.write_text("", encoding="utf-8")
    sample_path.write_text("", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = get_task(task_id)
    if max_steps_per_episode <= 0:
        max_steps = task.max_steps
    else:
        max_steps = int(max_steps_per_episode)

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1.4e-5, eps=1e-5)

    cost_agent = RuleBasedCostAgent()
    latency_agent = LatencyAgent()
    reason_samples: List[str] = []

    global_step = 0
    for ep in range(1, episodes + 1):
        env = ACREEnv(episode_length=task.episode_length, seed=seed + ep)
        state, info = env.reset(seed=seed + ep)
        done = False
        step_idx = 0
        while not done and step_idx < max_steps:
            step_idx += 1
            c_act = int(cost_agent.suggest_action(info))
            l_act = latency_agent.suggest_action(info)
            l_idx = int(l_act) if l_act is not None else None
            prompt = _build_prompt_text(tokenizer, info, c_act, l_idx)
            query = _encode_query(tokenizer, prompt, device)
            model.eval()
            with torch.inference_mode():
                g2d = model.pretrained_model.generate(
                    query.unsqueeze(0),
                    max_new_tokens=96,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            raw_full = g2d[0]
            qn = int(query.shape[0])
            # Match legacy TRL PPOTrainer(..., return_prompt=False): only newly generated token ids
            if raw_full.shape[0] < qn:
                resp = raw_full
            else:
                resp = raw_full[qn:]
            text = tokenizer.decode(resp, skip_special_tokens=True)
            action, reason, _ok = parse_action_json(text)
            idx = _action_name_to_idx(action)
            if len(reason_samples) < 20:
                line = f"ep={ep} step={step_idx} action={action} reason={reason[:240]!r} raw={text[:400]!r}\n"
                reason_samples.append(line)
                with sample_path.open("a", encoding="utf-8") as f:
                    f.write(line)

            next_s, r_env, term, trunc, ninfo = env.step(
                idx,
                cost_action=c_act,
                latency_action=l_idx,
            )
            r_combined = float(
                combined_step_reward(float(r_env), reason, action, normalize=True),
            )
            stats = _ppo_value_head_ministep(
                model,
                query,
                resp,
                float(r_combined),
                optim,
                device,
                cliprange=0.2,
                vf_coef=0.5,
            )

            row = {
                "episode": ep,
                "step": step_idx,
                "env_reward": float(r_env),
                "shaped_reward": r_combined,
                "action": action,
                "parse_ok": _ok,
                "cost": float(ninfo.get("step_cost", 0.0)),
                "latency_ms": float(ninfo.get("latency_ms", 0.0)),
            }
            if stats and isinstance(stats, dict):
                for k, v in stats.items():
                    if isinstance(v, (float, int, str, bool, type(None))):
                        row[f"stat_{k}"] = v
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            state = int(next_s)
            info = ninfo
            done = bool(term or trunc)
            global_step += 1

    save_path = output_dir / "model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    (output_dir / "training_meta.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "episodes": episodes,
                "model_name": model_name,
                "seed": seed,
                "output_dir": str(output_dir),
                "global_step": global_step,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Bonus: compare random vs LLM (same shaped reward not applied to random; env-only for apples-to-apples on metrics)
    llm = SRELLMAgent.from_checkpoint_dir(save_path, device="cuda" if torch.cuda.is_available() else "cpu")
    llm_res = evaluate_llm(task, seed=seed + 99_000, episodes=3, llm_agent=llm)
    rand_res = evaluate_random(task, seed=seed + 100_000, episodes=3)

    return {
        "model_path": str(save_path),
        "log_path": str(log_path),
        "reasoning_sample_path": str(sample_path),
        "random_eval_3ep": rand_res,
        "llm_eval_3ep": llm_res,
    }
