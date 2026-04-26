from __future__ import annotations

from typing import Dict, Optional


class LatencyAgent:
    """
    Lightweight UX-style agent.

    If latency is high, it recommends scaling up to improve user experience.
    """

    def __init__(self, threshold_ms: float = 220.0) -> None:
        self.threshold_ms = threshold_ms

    def suggest_action(self, info: Dict) -> Optional[int]:
        latency_ms = float(info.get("latency_ms", 0.0))
        if latency_ms > self.threshold_ms:
            return 1  # scale_up
        # No UX escalation recommendation on this step.
        return None
