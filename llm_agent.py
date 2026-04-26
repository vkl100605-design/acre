from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cost_agent import RuleBasedCostAgent
from env import ACTION_NAMES
from latency_agent import LatencyAgent
from parser import parse_action_json


def _traffic_level(demand: float) -> str:
    if demand < 0.4:
        return "low"
    if demand < 0.7:
        return "medium"
    return "high"


def _format_agent_action(idx: Optional[int]) -> str:
    if idx is None:
        return "none"
    return ACTION_NAMES.get(int(idx), "hold")


def build_sre_messages(
    info: Dict[str, Any],
    *,
    cost_suggestion: int,
    latency_suggestion: Optional[int],
) -> list[dict[str, str]]:
    load_ratio = float(info.get("load_ratio", 0.0))
    health = int(info.get("health", 2))
    health_s = {2: "healthy", 1: "degraded", 0: "down"}.get(health, "unknown")
    traffic = _traffic_level(float(info.get("traffic_demand", 0.0)))
    cost = float(info.get("step_cost", 0.0))
    latency = float(info.get("latency_ms", 0.0))
    err = float(info.get("error_rate", 0.0))
    raw_ct = str(info.get("conflict_type", "none") or "none")
    if raw_ct in ("", "none"):
        ct = "none"
    elif "latency" in raw_ct and "cost" in raw_ct:
        ct = "latency_vs_cost"
    else:
        ct = "latency_vs_cost" if (bool(info.get("overridden_by_cost_agent")) and bool(
            info.get("overridden_by_latency_agent")
        )) else "none"

    user_block = f"""You are managing a cloud system.

The current load is {load_ratio:.2f}x normal capacity.
System health score is {health} ({health_s}).
Error rate is {err:.2f}.
Latency is {latency:.1f} ms.
Current step cost is {cost:.2f}.
Traffic level is {traffic}.

Cost Agent Suggestion: {_format_agent_action(cost_suggestion)}
Latency Agent Suggestion: {_format_agent_action(latency_suggestion)}
Conflict context: {ct}

What action should be taken? Choose one of: hold, scale_up, scale_down, restart, rollback.

Respond ONLY in JSON:
{{
"action": "...",
"reason": "..."
}}
""".strip()

    return [
        {"role": "system", "content": "You are a senior SRE managing a cloud system."},
        {"role": "user", "content": user_block},
    ]


class SRELLMAgent:
    """
    Text policy: state (from info dict) -> (action name, reasoning).
    Uses a HF causal LM (can be a plain AutoModel; PPO will wrap value head in trl_training).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        *,
        cost_agent: Optional[RuleBasedCostAgent] = None,
        latency_agent: Optional[LatencyAgent] = None,
        max_new_tokens: int = 128,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cost_agent = cost_agent or RuleBasedCostAgent()
        self.latency_agent = latency_agent or LatencyAgent()
        self.max_new_tokens = int(max_new_tokens)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> "SRELLMAgent":
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        ).to(dev)
        model.eval()
        return cls(model, tokenizer, dev)

    @classmethod
    def from_checkpoint_dir(
        cls,
        path: Path,
        *,
        device: Optional[str] = None,
    ) -> "SRELLMAgent":
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        p = Path(path)
        torch_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(p, use_fast=True)
        try:
            from trl import AutoModelForCausalLMWithValueHead  # type: ignore
        except Exception:
            try:
                from trl.experimental.ppo import AutoModelForCausalLMWithValueHead  # type: ignore
            except Exception:
                AutoModelForCausalLMWithValueHead = None  # type: ignore[assignment, misc]
        if AutoModelForCausalLMWithValueHead is not None:
            try:
                model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    p,
                    torch_dtype=torch_dtype,
                ).to(dev)
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch_dtype).to(dev)
        else:
            model = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch_dtype).to(dev)
        model.eval()
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(model, tokenizer, dev)

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback: concat
        return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    @torch.inference_mode()
    def choose_action(self, info: Dict[str, Any]) -> Tuple[str, str]:
        cost_suggestion = int(self.cost_agent.suggest_action(info))
        latency_suggestion = self.latency_agent.suggest_action(info)
        if latency_suggestion is not None:
            latency_suggestion = int(latency_suggestion)

        messages = build_sre_messages(info, cost_suggestion=cost_suggestion, latency_suggestion=latency_suggestion)
        prompt = self._render_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        in_len = inputs["input_ids"].shape[1]
        gen_ids = out[0, in_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        action, reason, ok = parse_action_json(text)
        if not ok and action == "hold":
            # one retry with lower temperature
            out2 = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            text2 = self.tokenizer.decode(out2[0, in_len:], skip_special_tokens=True)
            action, reason, ok = parse_action_json(text2)
        return action, reason
