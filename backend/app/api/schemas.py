"""Pydantic request/response models.

Kept separate from the SQLAlchemy models and the domain dataclasses on
purpose. Three layers with three shapes:

- **Domain dataclasses** (`app.domain`) — pure business objects, no framework.
- **ORM models** (`app.db.models`) — persistence shape, includes physical
  concerns like partition keys.
- **API schemas** (here) — the public contract, versioned independently.

Collapsing these saves boilerplate and costs the ability to change the database
without breaking clients, or to evolve the API without a migration. For a
platform whose API is consumed by customer SDKs embedded in production
services, that decoupling is worth the extra file.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T")


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class PageParams(BaseModel):
    """Cursor pagination.

    Offset pagination is not offered on usage endpoints. At 10^9 rows,
    `OFFSET 5000000` forces Postgres to walk and discard five million rows, and
    the result is unstable — new events arriving between pages shift rows
    across page boundaries, so a client walking the list both re-sees and
    misses records. Keyset pagination on `(occurred_at, id)` is O(1) per page
    and stable under concurrent writes.
    """

    cursor: str | None = Field(default=None, description="Opaque cursor from the previous page")
    limit: int = Field(default=50, ge=1, le=500)


class Page(BaseModel, Generic[T]):
    items: list[T]
    next_cursor: str | None = None
    has_more: bool = False
    total_estimate: int | None = Field(
        default=None,
        description=(
            "Approximate row count from table statistics. Exact counts are not "
            "provided on usage endpoints because COUNT(*) over a partitioned "
            "fact table is a full scan."
        ),
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class TokenUsageIn(ApiModel):
    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cached_input: int = Field(default=0, ge=0)
    cache_write: int = Field(default=0, ge=0)
    reasoning: int = Field(default=0, ge=0)
    embedding: int = Field(default=0, ge=0)
    image: int = Field(default=0, ge=0)
    audio_input: int = Field(default=0, ge=0)
    audio_output: int = Field(default=0, ge=0)


class AttributionIn(ApiModel):
    department_id: UUID | None = None
    team_id: UUID | None = None
    user_id: UUID | None = None
    project: str | None = Field(default=None, max_length=120)
    feature: str | None = Field(default=None, max_length=120)
    application: str | None = Field(default=None, max_length=120)
    environment: str = Field(default="production", max_length=32)
    customer_id: str | None = Field(default=None, max_length=120)
    cost_center: str | None = Field(default=None, max_length=64)
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def _bound_tags(cls, value: dict[str, str]) -> dict[str, str]:
        """Cap tag cardinality.

        Unbounded tags are how a metrics system dies: a tag whose value is a
        request id creates a new time series per request. 20 keys is generous
        for legitimate use and cheap to enforce here rather than after the
        cardinality explosion has already been ingested.
        """
        if len(value) > 20:
            raise ValueError("at most 20 tags per event")
        for k, v in value.items():
            if len(k) > 40 or len(v) > 200:
                raise ValueError("tag keys must be <= 40 chars and values <= 200 chars")
        return value


class TraceIn(ApiModel):
    latency_ms: int = Field(default=0, ge=0)
    time_to_first_token_ms: int | None = Field(default=None, ge=0)
    streamed: bool = False
    retry_count: int = Field(default=0, ge=0)
    parent_request_id: str | None = Field(default=None, max_length=64)
    agent_run_id: str | None = Field(default=None, max_length=64)
    agent_step: int = Field(default=0, ge=0)
    conversation_id: str | None = Field(default=None, max_length=64)
    conversation_turn: int = Field(default=0, ge=0)
    rag_chunks: int = Field(default=0, ge=0)
    rag_tokens: int = Field(default=0, ge=0)
    system_prompt_tokens: int = Field(default=0, ge=0)
    few_shot_tokens: int = Field(default=0, ge=0)
    tool_definition_tokens: int = Field(default=0, ge=0)
    gpu_seconds: Decimal = Field(default=Decimal("0"), ge=0)
    network_gb: Decimal = Field(default=Decimal("0"), ge=0)


class UsageEventIn(ApiModel):
    provider: str
    model: str
    tokens: TokenUsageIn
    occurred_at: datetime | None = None
    idempotency_key: str | None = Field(default=None, max_length=128)
    model_type: str = "chat"
    status: str = "success"
    error_code: str | None = Field(default=None, max_length=64)
    attribution: AttributionIn = Field(default_factory=AttributionIn)
    trace: TraceIn = Field(default_factory=TraceIn)
    prompt_template_id: UUID | None = None
    prompt_version: str | None = Field(default=None, max_length=40)
    prompt_fingerprint: str | None = Field(default=None, max_length=64)
    served_from_semantic_cache: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestBatchIn(ApiModel):
    events: list[UsageEventIn] = Field(min_length=1, max_length=1000)


class IngestResult(ApiModel):
    accepted: int
    duplicates: int = 0
    rejected: int = 0
    unpriced: int = Field(
        default=0,
        description="Events accepted but not priced because no rate card matched. "
        "These raise an operational alert rather than being silently zero-costed.",
    )
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


class TimeWindow(BaseModel):
    start: datetime
    end: datetime

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, value: datetime, info) -> datetime:  # type: ignore[no-untyped-def]
        start = (info.data or {}).get("start")
        if start and value <= start:
            raise ValueError("end must be after start")
        return value


class CostSummaryOut(ApiModel):
    total_cost: Decimal
    token_cost: Decimal
    infrastructure_cost: Decimal
    cache_savings: Decimal
    wasted_cost: Decimal
    request_count: int
    total_tokens: int
    avg_cost_per_request: Decimal
    cost_per_1k_tokens: Decimal
    waste_ratio_pct: Decimal
    attribution_coverage_pct: Decimal
    period_over_period_pct: Decimal | None = None


class DimensionSliceOut(ApiModel):
    key: str
    label: str | None = None
    requests: int
    total_tokens: int
    cost: Decimal
    wasted_cost: Decimal
    cache_savings: Decimal
    avg_cost_per_request: Decimal
    avg_latency_ms: Decimal
    share_pct: Decimal


class TimeSeriesPoint(ApiModel):
    at: date
    cost: Decimal
    tokens: int
    requests: int


class TokenAnalyticsOut(ApiModel):
    prompt_p50: int
    prompt_p95: int
    prompt_p99: int
    completion_p50: int
    completion_p95: int
    context_p95: int
    composition: dict[str, float]
    static_share_pct: Decimal
    context_growth_per_turn: Decimal
    window_utilisation_pct: Decimal


class HeatmapOut(ApiModel):
    rows: list[str]
    columns: list[str]
    values: list[list[float]]


class SankeyLink(ApiModel):
    source: str
    target: str
    value: float


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


class ForecastPointOut(ApiModel):
    at: date
    value: Decimal
    lower: Decimal
    upper: Decimal


class ForecastOut(ApiModel):
    points: list[ForecastPointOut]
    method: str
    mape: Decimal | None = Field(
        default=None,
        description="Backtested mean absolute percentage error. Null when history "
        "is too short to backtest — treat such forecasts as directional only.",
    )
    confidence: Decimal
    seasonal: bool
    warnings: list[str] = Field(default_factory=list)
    horizon_totals: dict[str, Decimal] = Field(default_factory=dict)


class BudgetExhaustionOut(ApiModel):
    will_exhaust: bool
    exhausted_on: date | None
    days_remaining: int | None
    projected_period_spend: Decimal
    budget_amount: Decimal
    projected_overrun: Decimal
    burn_rate_per_day: Decimal
    utilisation_pct: Decimal


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------


class BudgetIn(ApiModel):
    name: str = Field(max_length=160)
    scope: Literal["organization", "department", "team", "user", "project", "feature", "model", "provider"]
    scope_id: str = Field(max_length=120)
    amount: Decimal = Field(gt=0)
    period: Literal["daily", "weekly", "monthly", "quarterly", "annual"] = "monthly"
    alert_thresholds: list[Decimal] = Field(
        default_factory=lambda: [Decimal("0.5"), Decimal("0.8"), Decimal("0.95")]
    )
    action_at_limit: Literal[
        "allow", "warn", "require_approval", "downgrade_model", "throttle", "block"
    ] = "warn"
    hard_stop_multiplier: Decimal | None = Field(default=None, gt=1)
    rollover: bool = False
    owner_email: str | None = None


class BudgetStatusOut(ApiModel):
    id: UUID
    name: str
    scope: str
    scope_id: str
    amount: Decimal
    spent: Decimal
    remaining: Decimal
    utilisation_pct: Decimal
    period: str
    period_start: date
    period_end: date
    severity: str
    is_exceeded: bool
    projected_to_exceed: bool
    forecast_period_spend: Decimal | None = None


class PreflightIn(ApiModel):
    """Pre-flight check payload. Must stay small — this is on the hot path."""

    provider: str
    model: str
    estimated_input_tokens: int = Field(ge=0)
    estimated_output_tokens: int = Field(ge=0)
    department_id: str | None = None
    team_id: str | None = None
    user_id: str | None = None
    feature: str | None = None
    environment: str = "production"


class PreflightOut(ApiModel):
    action: str
    allowed: bool
    estimated_cost: Decimal
    reasons: list[str] = Field(default_factory=list)
    violated_policies: list[str] = Field(default_factory=list)
    suggested_model: str | None = None
    evaluated_in_ms: float


class ChargebackLineOut(ApiModel):
    cost_center: str
    department: str | None
    direct_cost: Decimal
    allocated_shared_cost: Decimal
    total: Decimal
    tokens: int
    requests: int
    share_pct: Decimal


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


class RecommendationOut(ApiModel):
    id: UUID
    kind: str
    title: str
    rationale: str
    scope: str
    scope_key: str
    estimated_monthly_savings: Decimal
    annual_savings: Decimal
    confidence: Decimal
    priority_score: Decimal
    severity: str
    effort: str
    implementation_hours: Decimal
    risk: str
    quality_impact: Decimal
    requires_evaluation: bool
    blocked_by_quality: bool
    quality_note: str | None = None
    implementation_steps: list[str]
    evidence: dict[str, Any]
    status: str


class PromptAnalysisIn(ApiModel):
    text: str = Field(max_length=500_000)
    exact_token_count: int | None = Field(default=None, ge=0)
    static_prefix_chars: int = Field(default=0, ge=0)
    monthly_calls: int = Field(default=1000, ge=0)
    rate_per_token: Decimal = Field(default=Decimal("0.0000025"), ge=0)


class PromptFindingOut(ApiModel):
    kind: str
    detail: str
    tokens_saved: int
    confidence: float
    span: list[int] | None = None
    severity: str


class PromptAnalysisOut(ApiModel):
    fingerprint: str
    total_tokens: int
    recoverable_tokens: int
    compression_ratio_pct: Decimal
    cacheable_ratio_pct: Decimal
    efficiency_score: Decimal
    monthly_savings: Decimal
    annual_savings: Decimal
    quality_risk: str
    findings: list[PromptFindingOut]


class RoutingIn(ApiModel):
    expected_input_tokens: int = Field(ge=0)
    expected_output_tokens: int = Field(ge=0)
    objective: Literal[
        "cheapest", "fastest", "highest_quality", "balanced", "reasoning", "vision",
        "embedding", "local_only",
    ] = "balanced"
    complexity: Literal["trivial", "simple", "moderate", "complex", "expert"] = "moderate"
    baseline_model: str | None = None
    requires_vision: bool = False
    requires_tools: bool = False
    max_latency_ms: int | None = Field(default=None, gt=0)
    allowed_providers: list[str] | None = None
    require_self_hosted: bool = False


class RoutingCandidateOut(ApiModel):
    provider: str
    model: str
    estimated_cost: Decimal
    estimated_latency_ms: int
    quality: Decimal
    score: Decimal
    factors: dict[str, float]


class RoutingOut(ApiModel):
    selected: RoutingCandidateOut | None
    alternatives: list[RoutingCandidateOut]
    baseline: RoutingCandidateOut | None
    savings_vs_baseline: Decimal
    savings_pct: Decimal
    quality_delta: Decimal
    rationale: str
    rejected_count: int
    rejected_sample: dict[str, str] = Field(default_factory=dict)


class SimulationLeverIn(ApiModel):
    type: Literal[
        "switch_model", "compress_prompt", "enable_prompt_cache", "enable_response_cache",
        "reduce_rag_context", "summarise_history", "batch_requests",
    ]
    provider: str | None = None
    model: str | None = None
    reduction_pct: Decimal | None = Field(default=None, ge=0, le=100)
    hit_rate: Decimal | None = Field(default=None, ge=0, le=1)
    quality_delta: Decimal | None = None
    eligible_ratio: Decimal | None = Field(default=None, ge=0, le=1)


class WorkloadProfileIn(ApiModel):
    provider: str
    model: str
    monthly_requests: int = Field(ge=0)
    avg_input_tokens: int = Field(ge=0)
    avg_output_tokens: int = Field(ge=0)
    avg_cached_input_tokens: int = Field(default=0, ge=0)
    avg_reasoning_tokens: int = Field(default=0, ge=0)
    static_input_tokens: int = Field(default=0, ge=0)
    rag_input_tokens: int = Field(default=0, ge=0)
    quality_index: Decimal = Field(default=Decimal("0.90"), ge=0, le=1)
    cacheable_request_ratio: Decimal = Field(default=Decimal("0"), ge=0, le=1)


class SimulationIn(ApiModel):
    profile: WorkloadProfileIn
    levers: list[SimulationLeverIn] = Field(default_factory=list)
    scenario_name: str = "scenario"
    include_standard_scenarios: bool = False


class ScenarioOut(ApiModel):
    name: str
    baseline_monthly_cost: Decimal
    projected_monthly_cost: Decimal
    monthly_savings: Decimal
    annual_savings: Decimal
    savings_pct: Decimal
    token_reduction_pct: Decimal
    quality_delta: Decimal
    latency_multiplier: Decimal
    levers: list[str]
    notes: list[str]
    warnings: list[str]
    recommendation: str


class AnomalyOut(ApiModel):
    id: UUID | None = None
    kind: str
    severity: str
    title: str
    detail: str
    scope: str
    scope_key: str | None
    observed_value: Decimal
    expected_value: Decimal
    deviation_score: Decimal
    estimated_impact: Decimal
    evidence: dict[str, Any]
    recommended_action: str | None
    detected_at: datetime
    is_resolved: bool = False


class RagConfigIn(ApiModel):
    chunk_size: int = Field(default=512, ge=32, le=8192)
    chunk_overlap: int = Field(default=64, ge=0, le=2048)
    top_k: int = Field(default=10, ge=1, le=100)
    reranker_enabled: bool = False
    hybrid_search: bool = False
    embedding_model: str = "text-embedding-3-small"
    monthly_requests: int = Field(default=10_000, ge=0)
    input_rate_per_token: Decimal = Field(default=Decimal("0.0000025"), ge=0)
    avg_answer_span_tokens: int = Field(default=180, ge=10)


class RagOptimizationOut(ApiModel):
    lever: str
    current: str
    proposed: str
    tokens_saved_per_request: int
    monthly_savings: Decimal
    estimated_recall_delta: Decimal
    quality_risk: str
    confidence: Decimal
    requires_evaluation: bool
    rationale: str


class RagAdvisoryOut(ApiModel):
    current_context_tokens_per_request: int
    overlap_waste_tokens_per_request: int
    total_monthly_savings: Decimal
    safe_savings_no_evaluation_needed: Decimal
    optimizations: list[RagOptimizationOut]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class LoginIn(ApiModel):
    email: str
    password: str


class TokenOut(ApiModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class MeOut(ApiModel):
    id: UUID
    email: str
    full_name: str | None
    role: str
    organization_id: UUID
    permissions: list[str]


class ErrorOut(ApiModel):
    """RFC 7807-style problem detail.

    A machine-readable `code` alongside the human `detail` so SDKs can branch
    on failure type without string-matching English prose that may be
    reworded.
    """

    code: str
    detail: str
    request_id: str | None = None
    fields: dict[str, str] | None = None
