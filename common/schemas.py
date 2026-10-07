"""Shared request/response contracts (Pydantic) used by triage-api and
scoring-service, so the two never disagree on the wire format."""
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PaymentType(StrEnum):
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"


class PaymentRequest(BaseModel):
    """One payment to be triaged, submitted by the upstream payment app."""

    txn_id: str = Field(min_length=1, max_length=64)
    step: int = Field(ge=0, description="Hour index since epoch of the batch window")
    type: PaymentType
    amount: float = Field(gt=0, le=1_000_000_000)
    name_orig: str = Field(min_length=1, alias="nameOrig")
    oldbalance_orig: float = Field(ge=0, alias="oldbalanceOrg")
    name_dest: str = Field(min_length=1, alias="nameDest")
    oldbalance_dest: float = Field(ge=0, alias="oldbalanceDest")

    model_config = {"populate_by_name": True}

    @field_validator("name_orig", "name_dest")
    @classmethod
    def strip(cls, v: str) -> str:
        return v.strip()


class Decision(StrEnum):
    ALLOW = "ALLOW"
    STEP_UP = "STEP_UP"
    BLOCK = "BLOCK"


class ScoreResult(BaseModel):
    """scoring-service's response: a pure risk assessment, no policy applied."""

    txn_id: str
    risk_score: float = Field(ge=0, le=1)
    reason_codes: list[str]
    model_name: str
    model_version: str
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TriageDecision(BaseModel):
    """triage-api's final response to the caller."""

    txn_id: str
    decision: Decision
    risk_score: float
    reason_codes: list[str]
    model_version: str
    degraded: bool = False
    rule_triggered: str | None = None
    latency_ms: float


class PaymentDecidedEvent(BaseModel):
    """Published to the payment.decided stream for downstream consumers."""

    txn_id: str
    decision: Decision
    risk_score: float
    reason_codes: list[str]
    amount: float
    name_orig: str
    name_dest: str
    step: int
    model_version: str
    degraded: bool
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Case(BaseModel):
    case_id: str
    txn_id: str
    priority: float
    amount: float
    risk_score: float
    decision: Decision
    reason_codes: list[str]
    status: Literal["OPEN", "RESOLVED"] = "OPEN"
    verdict: Literal["CONFIRMED_FRAUD", "GENUINE"] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None
