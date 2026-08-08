"""The usage event — the platform's atomic fact.

Every downstream number (dashboards, forecasts, chargeback, anomaly scores)
is an aggregation over these. Two design decisions dominate:

1. **Events are immutable and idempotent.** Each carries a client-supplied
   `idempotency_key`; the ingestion path upserts on it. SDKs retry on network
   failure, and a retried event that double-counts cost is worse than a lost
   one — finance can explain a gap, but not a phantom charge.

2. **Attribution is captured at emit time, not inferred later.** The SDK
   stamps department/team/feature/project onto the event. Reconstructing "who
   did this" after the fact from IP addresses or API keys is where cost
   attribution projects usually die: keys get shared, services get renamed,
   and the mapping is unrecoverable six months on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID, uuid4

from app.domain.enums import (
    BILLABLE_STATUSES,
    ModelType,
    Provider,
    RequestStatus,
    TokenClass,
)
from app.domain.money import ZERO, to_decimal


@dataclass
class TokenUsage:
    """Token counts broken out by billable class.

    Note `cached_input` is *not* a subset of `input`: providers report them as
    disjoint buckets (tokens served from cache vs. freshly processed), and
    treating cached tokens as a discount on the input count rather than a
    separate cheaper class produces cost errors of up to 90% on
    cache-heavy workloads.
    """

    input: int = 0
    output: int = 0
    cached_input: int = 0
    cache_write: int = 0
    reasoning: int = 0
    embedding: int = 0
    image: int = 0
    audio_input: int = 0
    audio_output: int = 0

    def as_map(self) -> dict[TokenClass, int]:
        return {
            TokenClass.INPUT: self.input,
            TokenClass.OUTPUT: self.output,
            TokenClass.CACHED_INPUT: self.cached_input,
            TokenClass.CACHE_WRITE: self.cache_write,
            TokenClass.REASONING: self.reasoning,
            TokenClass.EMBEDDING: self.embedding,
            TokenClass.IMAGE: self.image,
            TokenClass.AUDIO_INPUT: self.audio_input,
            TokenClass.AUDIO_OUTPUT: self.audio_output,
        }

    @property
    def total(self) -> int:
        return sum(self.as_map().values())

    @property
    def prompt_side(self) -> int:
        """Everything the model had to read."""
        return self.input + self.cached_input + self.cache_write + self.embedding + self.audio_input

    @property
    def completion_side(self) -> int:
        """Everything the model produced, including hidden reasoning."""
        return self.output + self.reasoning + self.image + self.audio_output

    @property
    def cache_hit_ratio(self) -> Decimal:
        readable = self.input + self.cached_input
        if readable == 0:
            return ZERO
        return Decimal(self.cached_input) / Decimal(readable)

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            cached_input=self.cached_input + other.cached_input,
            cache_write=self.cache_write + other.cache_write,
            reasoning=self.reasoning + other.reasoning,
            embedding=self.embedding + other.embedding,
            image=self.image + other.image,
            audio_input=self.audio_input + other.audio_input,
            audio_output=self.audio_output + other.audio_output,
        )


@dataclass
class AttributionContext:
    """The "who and why" dimensions of a request.

    All optional because instrumentation rolls out incrementally — a platform
    that rejects events lacking a cost centre gets bypassed, and unattributed
    spend that we can *see* is far more useful than spend we never received.
    Unattributed volume is surfaced as its own governance metric to drive
    instrumentation coverage up over time.
    """

    organization_id: Optional[UUID] = None
    department_id: Optional[UUID] = None
    team_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    project: Optional[str] = None
    feature: Optional[str] = None
    environment: str = "production"
    application: Optional[str] = None
    customer_id: Optional[str] = None
    cost_center: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def is_attributed(self) -> bool:
        return any((self.department_id, self.team_id, self.cost_center))


@dataclass
class RequestTrace:
    """Execution characteristics used for quality, latency and waste analysis."""

    latency_ms: int = 0
    time_to_first_token_ms: Optional[int] = None
    streamed: bool = False
    retry_count: int = 0
    #: Set when this call is a retry of a prior failed attempt; lets the waste
    #: detector distinguish "expensive workload" from "expensive flailing".
    parent_request_id: Optional[str] = None
    #: Depth in an agent loop. Runaway agents are detected on this.
    agent_step: int = 0
    agent_run_id: Optional[str] = None
    conversation_id: Optional[str] = None
    conversation_turn: int = 0
    #: Number of RAG chunks injected, and their token weight.
    rag_chunks: int = 0
    rag_tokens: int = 0
    system_prompt_tokens: int = 0
    few_shot_tokens: int = 0
    tool_definition_tokens: int = 0
    #: Self-hosted only.
    gpu_seconds: Decimal = ZERO
    network_gb: Decimal = ZERO


@dataclass
class UsageEvent:
    """One metered interaction with a model."""

    provider: Provider
    model: str
    tokens: TokenUsage
    id: UUID = field(default_factory=uuid4)
    idempotency_key: Optional[str] = None
    model_type: ModelType = ModelType.CHAT
    status: RequestStatus = RequestStatus.SUCCESS
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attribution: AttributionContext = field(default_factory=AttributionContext)
    trace: RequestTrace = field(default_factory=RequestTrace)
    prompt_template_id: Optional[UUID] = None
    prompt_version: Optional[str] = None
    prompt_fingerprint: Optional[str] = None
    #: Whether the platform's own semantic cache served this — distinct from
    #: the provider's prompt cache, which shows up in `tokens.cached_input`.
    served_from_semantic_cache: bool = False
    error_code: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_billable(self) -> bool:
        return self.status in BILLABLE_STATUSES

    @property
    def is_wasted(self) -> bool:
        """Spend that produced no business value.

        Retries, hard errors and cancellations all consumed provider capacity
        without delivering an answer to a user. Quantifying this is usually
        the fastest credible win in a cost programme, because nobody defends
        a line item labelled "money spent on failures".
        """
        if self.status in {RequestStatus.ERROR, RequestStatus.TIMEOUT}:
            return True
        return self.trace.retry_count > 0 and self.trace.parent_request_id is not None


@dataclass
class CostBreakdown:
    """Fully-resolved cost of one usage event.

    Kept as a separate object from the event so that re-pricing (a corrected
    rate card, a retroactive discount) can regenerate costs without touching
    the immutable usage record.
    """

    event_id: UUID
    currency: str = "USD"
    by_token_class: dict[TokenClass, Decimal] = field(default_factory=dict)
    request_fee: Decimal = ZERO
    compute_cost: Decimal = ZERO
    energy_cost: Decimal = ZERO
    network_cost: Decimal = ZERO
    storage_cost: Decimal = ZERO
    #: What this call *would* have cost with no cache hits. The delta against
    #: `total` is the realised saving, which is what funds the caching
    #: programme in front of a CFO.
    uncached_equivalent: Decimal = ZERO
    rate_card_version: Optional[str] = None
    priced_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def token_cost(self) -> Decimal:
        return sum(self.by_token_class.values(), ZERO)

    @property
    def infrastructure_cost(self) -> Decimal:
        return self.compute_cost + self.energy_cost + self.network_cost + self.storage_cost

    @property
    def total(self) -> Decimal:
        return self.token_cost + self.request_fee + self.infrastructure_cost

    @property
    def cache_savings(self) -> Decimal:
        saving = self.uncached_equivalent - self.total
        return saving if saving > ZERO else ZERO


@dataclass
class UsageAggregate:
    """Rolled-up usage over a dimension slice, as served to dashboards."""

    key: str
    tokens: TokenUsage = field(default_factory=TokenUsage)
    cost: Decimal = ZERO
    wasted_cost: Decimal = ZERO
    cache_savings: Decimal = ZERO
    request_count: int = 0
    error_count: int = 0
    retry_count: int = 0
    latency_ms_sum: int = 0

    @property
    def avg_latency_ms(self) -> Decimal:
        if self.request_count == 0:
            return ZERO
        return Decimal(self.latency_ms_sum) / Decimal(self.request_count)

    @property
    def cost_per_request(self) -> Decimal:
        if self.request_count == 0:
            return ZERO
        return self.cost / Decimal(self.request_count)

    @property
    def cost_per_1k_tokens(self) -> Decimal:
        if self.tokens.total == 0:
            return ZERO
        return (self.cost / Decimal(self.tokens.total)) * Decimal("1000")

    @property
    def error_rate(self) -> Decimal:
        if self.request_count == 0:
            return ZERO
        return Decimal(self.error_count) / Decimal(self.request_count)

    def absorb(self, event: UsageEvent, cost: CostBreakdown) -> None:
        self.tokens = self.tokens + event.tokens
        self.cost += cost.total
        self.cache_savings += cost.cache_savings
        self.request_count += 1
        self.latency_ms_sum += event.trace.latency_ms
        self.retry_count += event.trace.retry_count
        if event.status is not RequestStatus.SUCCESS:
            self.error_count += 1
        if event.is_wasted:
            self.wasted_cost += cost.total


def coerce_tokens(raw: dict[str, Any] | TokenUsage | None) -> TokenUsage:
    """Build a TokenUsage from loosely-shaped SDK payloads.

    Provider SDKs disagree on field names for the same quantity
    (`prompt_tokens` vs `input_tokens`, `completion_tokens` vs
    `output_tokens`). Normalising here — rather than in each collector — means
    a new provider integration only has to map into one vocabulary.
    """
    if isinstance(raw, TokenUsage):
        return raw
    if not raw:
        return TokenUsage()

    def pick(*names: str) -> int:
        for name in names:
            value = raw.get(name)
            if value is not None:
                return int(to_decimal(value))
        return 0

    return TokenUsage(
        input=pick("input", "input_tokens", "prompt_tokens"),
        output=pick("output", "output_tokens", "completion_tokens"),
        cached_input=pick("cached_input", "cache_read_input_tokens", "cached_tokens"),
        cache_write=pick("cache_write", "cache_creation_input_tokens"),
        reasoning=pick("reasoning", "reasoning_tokens", "thoughts_token_count"),
        embedding=pick("embedding", "embedding_tokens"),
        image=pick("image", "image_tokens"),
        audio_input=pick("audio_input", "audio_input_tokens"),
        audio_output=pick("audio_output", "audio_output_tokens"),
    )
