"""Triage policy (GR4ML PrescriptionGoal, implemented): maps a risk score to
one of ALLOW / STEP_UP / BLOCK using the cost-optimal thresholds chosen
offline by training/select_thresholds.py. Thresholds are data, not code, so
they can be updated without a redeploy."""
import json
from pathlib import Path

from common.schemas import Decision


class TriagePolicy:
    def __init__(self, thresholds_path: str) -> None:
        data = json.loads(Path(thresholds_path).read_text())
        self.t_stepup = data["t_stepup"]
        self.t_block = data["t_block"]

    def decide(self, risk_score: float) -> Decision:
        if risk_score >= self.t_block:
            return Decision.BLOCK
        if risk_score >= self.t_stepup:
            return Decision.STEP_UP
        return Decision.ALLOW
