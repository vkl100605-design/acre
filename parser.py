from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

_VALID_ACTIONS = {"hold", "scale_up", "scale_down", "restart", "rollback"}


def parse_action_json(
    text: str,
) -> Tuple[str, str, bool]:
    """
    Extract {"action": "...", "reason": "..."} from model output.
    Returns (action, reason, ok). On failure, returns ("hold", <truncated text>, False).
    """
    raw = (text or "").strip()
    if not raw:
        return "hold", "", False

    match = re.search(r"\{[^{}]*\}", raw, flags=re.S)
    candidate = match.group(0) if match else raw

    try:
        data = json.loads(candidate)
    except Exception:
        try:
            # Allow single-quoted key/value JSON-ish fragments
            fixed = candidate.replace("'", '"')
            data = json.loads(fixed)
        except Exception:
            return "hold", raw[:200], False

    if not isinstance(data, dict):
        return "hold", raw[:200], False

    action = str(data.get("action", "hold")).strip().lower()
    reason = str(data.get("reason", "")).strip()
    if action not in _VALID_ACTIONS:
        return "hold", reason or raw[:200], False
    return action, reason, True
