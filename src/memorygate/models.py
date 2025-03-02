"""Pydantic data models: dataset, memory items and run results."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    speaker: Literal["user", "assistant"]
    text: str


class Conversation(BaseModel):
    id: str
    customer_id: str
    timestamp: int
    turns: list[Turn]


class StaleFact(BaseModel):
    """A value that was once stated but is no longer true."""

    attribute: str
    value: str
    reason: Literal["update", "contradiction"]


class Customer(BaseModel):
    id: str
    is_distractor: bool = False
    facts: dict[str, str]  # attribute -> current (ground-truth) value
    stale: list[StaleFact] = Field(default_factory=list)


class Query(BaseModel):
    customer_id: str
    attribute: str
    text: str
    expected_value: str


class GeneratorConfig(BaseModel):
    seed: int = 7
    n_customers: int = Field(60, ge=2)
    distractor_ratio: float = Field(0.25, ge=0, le=2)
    min_facts: int = Field(4, ge=1)
    max_facts: int = Field(6, ge=1)
    update_prob: float = Field(0.5, ge=0, le=1)
    contradiction_prob: float = Field(0.3, ge=0, le=1)
    third_party_prob: float = Field(0.5, ge=0, le=1)
    hard_prob: float = Field(0.12, ge=0, le=1)


class Dataset(BaseModel):
    config: GeneratorConfig
    customers: list[Customer]
    conversations: list[Conversation]
    queries: list[Query]

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]

    def owners(self) -> dict[str, str]:
        """conversation id -> customer id (the ground truth for leakage)."""
        return {c.id: c.customer_id for c in self.conversations}

    def evaluated(self) -> list[Customer]:
        return [c for c in self.customers if not c.is_distractor]


class MemoryItem(BaseModel):
    """One memory returned by an adapter. `conversation_id` is its provenance."""

    attribute: str | None = None
    value: str
    conversation_id: str
    text: str = ""
    score: float = 0.0


class CustomerStats(BaseModel):
    """Raw per-customer counts; metrics are ratios of sums of these."""

    tp: float = 0
    fp: float = 0
    fn: float = 0
    stale_hits: float = 0
    stale_total: float = 0
    leaked: float = 0
    retrieved: float = 0
    hits: float = 0
    rr_sum: float = 0
    queries: float = 0


class RunResult(BaseModel):
    schema_version: int = 1
    adapter: str
    k: int
    dataset_config: GeneratorConfig
    dataset_fingerprint: str
    metrics: dict[str, float]
    per_customer: dict[str, CustomerStats]


class MetricComparison(BaseModel):
    name: str
    higher_is_better: bool
    baseline: float
    candidate: float
    delta: float
    delta_ci_low: float
    delta_ci_high: float
    worsening_bound: float  # one-sided lower confidence bound on the worsening
    status: Literal["regression", "improved", "ok"]


class GateReport(BaseModel):
    passed: bool
    baseline_adapter: str
    candidate_adapter: str
    k: int
    n_customers: int
    confidence: float
    min_effect: float
    n_bootstrap: int
    metrics: list[MetricComparison]
