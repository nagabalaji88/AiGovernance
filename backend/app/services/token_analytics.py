"""Token analytics: where the tokens actually go.

Cost dashboards tell a team *that* they spent $40k. This module tells them
*which part of the prompt* spent it — system preamble, few-shot examples, tool
definitions, RAG context, or conversation history. That decomposition is the
difference between "reduce your spend" (unactionable) and "your 4,000-token
system prompt is 71% of every call and hasn't changed in six months, so cache
it" (a one-line fix).

All functions are pure and operate on lists of events already in memory. The
callers are (a) the aggregation worker, which streams a partition at a time,
and (b) the API, which passes a bounded, pre-filtered window.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import pairwise
from statistics import median

from app.domain.money import ZERO, safe_div, to_decimal
from app.domain.usage import CostBreakdown, UsageEvent

#: Prompt regions we decompose input tokens into. `context_other` is the
#: residual — a large residual means instrumentation gaps, and we surface it
#: rather than silently folding it into one of the named buckets.
PROMPT_REGIONS = (
    "system_prompt",
    "few_shot",
    "tool_definitions",
    "rag_context",
    "conversation_history",
    "user_message",
)


@dataclass(slots=True)
class TokenDistribution:
    """Percentile view of a token count population.

    Means are actively misleading for token distributions: they are heavily
    right-skewed (most calls small, a long tail of enormous context dumps), so
    the mean sits above the 70th percentile and describes nobody's experience.
    p50/p95/p99 is what capacity and cost planning actually need.
    """

    count: int = 0
    total: int = 0
    p50: int = 0
    p90: int = 0
    p95: int = 0
    p99: int = 0
    maximum: int = 0
    mean: Decimal = ZERO

    @classmethod
    def from_values(cls, values: list[int]) -> TokenDistribution:
        if not values:
            return cls()
        ordered = sorted(values)
        n = len(ordered)

        def pct(p: float) -> int:
            # Nearest-rank percentile: unambiguous, matches what an operator
            # gets from `sort | head -n`, and needs no interpolation policy.
            idx = min(n - 1, max(0, round(p * n) - 1))
            return ordered[idx]

        total = sum(ordered)
        return cls(
            count=n,
            total=total,
            p50=int(median(ordered)),
            p90=pct(0.90),
            p95=pct(0.95),
            p99=pct(0.99),
            maximum=ordered[-1],
            mean=safe_div(Decimal(total), Decimal(n)),
        )


@dataclass(slots=True)
class PromptComposition:
    """Average decomposition of the prompt side of a call."""

    system_prompt: int = 0
    few_shot: int = 0
    tool_definitions: int = 0
    rag_context: int = 0
    conversation_history: int = 0
    user_message: int = 0
    context_other: int = 0

    @property
    def total(self) -> int:
        return (
            self.system_prompt
            + self.few_shot
            + self.tool_definitions
            + self.rag_context
            + self.conversation_history
            + self.user_message
            + self.context_other
        )

    def share(self, region: str) -> Decimal:
        total = self.total
        if total == 0:
            return ZERO
        return safe_div(Decimal(getattr(self, region, 0)), Decimal(total)) * Decimal("100")

    def as_shares(self) -> dict[str, Decimal]:
        regions = (*PROMPT_REGIONS, "context_other")
        return {r: self.share(r) for r in regions}


@dataclass(slots=True)
class ContextEfficiency:
    """How well a workload uses the context window it pays for."""

    avg_context_tokens: Decimal = ZERO
    p95_context_tokens: int = 0
    window_size: int = 0
    #: Fraction of the paid-for window actually filled at p95.
    utilisation: Decimal = ZERO
    #: Share of prompt tokens that recur unchanged call-to-call. High values
    #: are the strongest possible signal for prompt caching.
    static_share: Decimal = ZERO
    #: Growth in context tokens per additional conversation turn.
    growth_per_turn: Decimal = ZERO


@dataclass(slots=True)
class DimensionSlice:
    """Aggregated metrics for one value of a grouping dimension."""

    key: str
    requests: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: Decimal = ZERO
    wasted_cost: Decimal = ZERO
    cache_savings: Decimal = ZERO
    latency_ms_sum: int = 0

    @property
    def cost_share_basis(self) -> Decimal:
        return self.cost

    @property
    def avg_tokens_per_request(self) -> Decimal:
        return safe_div(Decimal(self.total_tokens), Decimal(self.requests))

    @property
    def avg_cost_per_request(self) -> Decimal:
        return safe_div(self.cost, Decimal(self.requests))

    @property
    def avg_latency_ms(self) -> Decimal:
        return safe_div(Decimal(self.latency_ms_sum), Decimal(self.requests))

    @property
    def output_ratio(self) -> Decimal:
        return safe_div(Decimal(self.completion_tokens), Decimal(self.total_tokens))


@dataclass(slots=True)
class TokenAnalyticsReport:
    window_start: str | None = None
    window_end: str | None = None
    prompt_distribution: TokenDistribution = field(default_factory=TokenDistribution)
    completion_distribution: TokenDistribution = field(default_factory=TokenDistribution)
    context_distribution: TokenDistribution = field(default_factory=TokenDistribution)
    composition: PromptComposition = field(default_factory=PromptComposition)
    context_efficiency: ContextEfficiency = field(default_factory=ContextEfficiency)
    total_cost: Decimal = ZERO
    wasted_cost: Decimal = ZERO
    cache_savings: Decimal = ZERO
    unattributed_cost: Decimal = ZERO
    slices: dict[str, list[DimensionSlice]] = field(default_factory=dict)

    @property
    def waste_ratio(self) -> Decimal:
        return safe_div(self.wasted_cost, self.total_cost) * Decimal("100")

    @property
    def attribution_coverage(self) -> Decimal:
        """Share of spend that carries a cost centre. A governance KPI in its
        own right — you cannot chargeback what you cannot attribute."""
        if self.total_cost == ZERO:
            return Decimal("100")
        return (Decimal("1") - safe_div(self.unattributed_cost, self.total_cost)) * Decimal("100")


#: Grouping functions for the standard dimension set. Adding a dimension is a
#: one-line change here plus a UI facet — the aggregation code is generic.
DIMENSION_EXTRACTORS: dict[str, Callable[[UsageEvent], str | None]] = {
    "provider": lambda e: str(e.provider),
    "model": lambda e: f"{e.provider}/{e.model}",
    "model_type": lambda e: str(e.model_type),
    "department": lambda e: str(e.attribution.department_id) if e.attribution.department_id else None,
    "team": lambda e: str(e.attribution.team_id) if e.attribution.team_id else None,
    "user": lambda e: str(e.attribution.user_id) if e.attribution.user_id else None,
    "feature": lambda e: e.attribution.feature,
    "project": lambda e: e.attribution.project,
    "application": lambda e: e.attribution.application,
    "environment": lambda e: e.attribution.environment,
    "customer": lambda e: e.attribution.customer_id,
    "cost_center": lambda e: e.attribution.cost_center,
    "prompt_template": lambda e: str(e.prompt_template_id) if e.prompt_template_id else None,
    "status": lambda e: str(e.status),
}


def analyse(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown] | list[CostBreakdown],
    *,
    dimensions: Iterable[str] = ("provider", "model", "department", "team", "feature"),
    window_size: int = 0,
) -> TokenAnalyticsReport:
    """Build a full token analytics report over a window of priced events."""
    cost_by_id = _index_costs(costs)
    report = TokenAnalyticsReport()
    if not events:
        return report

    ordered = sorted(events, key=lambda e: e.occurred_at)
    report.window_start = ordered[0].occurred_at.isoformat()
    report.window_end = ordered[-1].occurred_at.isoformat()

    prompt_values: list[int] = []
    completion_values: list[int] = []
    context_values: list[int] = []
    composition = PromptComposition()
    slices: dict[str, dict[str, DimensionSlice]] = {d: {} for d in dimensions}
    turn_context: list[tuple[int, int]] = []
    static_tokens = 0
    prompt_total = 0

    for event in ordered:
        cost = cost_by_id.get(str(event.id))
        amount = cost.total if cost else ZERO
        savings = cost.cache_savings if cost else ZERO

        prompt = event.tokens.prompt_side
        completion = event.tokens.completion_side
        prompt_values.append(prompt)
        completion_values.append(completion)
        context_values.append(prompt + completion)

        t = event.trace
        composition.system_prompt += t.system_prompt_tokens
        composition.few_shot += t.few_shot_tokens
        composition.tool_definitions += t.tool_definition_tokens
        composition.rag_context += t.rag_tokens
        named = t.system_prompt_tokens + t.few_shot_tokens + t.tool_definition_tokens + t.rag_tokens
        # Whatever the SDK did not label is residual context. We split it
        # heuristically between history and the user's own message using turn
        # index: turn 0 has no history by definition.
        residual = max(0, prompt - named)
        if t.conversation_turn > 0:
            composition.conversation_history += int(residual * 0.7)
            composition.user_message += residual - int(residual * 0.7)
        else:
            composition.user_message += residual

        # System prompt + tool definitions + few-shot are the call-invariant
        # regions; they are what a prompt cache would eliminate.
        static_tokens += t.system_prompt_tokens + t.tool_definition_tokens + t.few_shot_tokens
        prompt_total += prompt

        if t.conversation_turn > 0:
            turn_context.append((t.conversation_turn, prompt))

        report.total_cost += amount
        report.cache_savings += savings
        if event.is_wasted:
            report.wasted_cost += amount
        if not event.attribution.is_attributed:
            report.unattributed_cost += amount

        for dim in dimensions:
            extractor = DIMENSION_EXTRACTORS.get(dim)
            if extractor is None:
                continue
            key = extractor(event) or "unattributed"
            bucket = slices[dim].get(key)
            if bucket is None:
                bucket = DimensionSlice(key=key)
                slices[dim][key] = bucket
            bucket.requests += 1
            bucket.total_tokens += event.tokens.total
            bucket.prompt_tokens += prompt
            bucket.completion_tokens += completion
            bucket.cost += amount
            bucket.cache_savings += savings
            bucket.latency_ms_sum += t.latency_ms
            if event.is_wasted:
                bucket.wasted_cost += amount

    report.prompt_distribution = TokenDistribution.from_values(prompt_values)
    report.completion_distribution = TokenDistribution.from_values(completion_values)
    report.context_distribution = TokenDistribution.from_values(context_values)
    report.composition = composition
    report.context_efficiency = _context_efficiency(
        report.context_distribution, static_tokens, prompt_total, turn_context, window_size
    )
    report.slices = {
        dim: sorted(buckets.values(), key=lambda s: s.cost, reverse=True) for dim, buckets in slices.items()
    }
    return report


def _context_efficiency(
    context: TokenDistribution,
    static_tokens: int,
    prompt_total: int,
    turn_context: list[tuple[int, int]],
    window_size: int,
) -> ContextEfficiency:
    eff = ContextEfficiency(
        avg_context_tokens=context.mean,
        p95_context_tokens=context.p95,
        window_size=window_size,
        static_share=safe_div(Decimal(static_tokens), Decimal(prompt_total)) * Decimal("100"),
    )
    if window_size > 0:
        eff.utilisation = safe_div(Decimal(context.p95), Decimal(window_size)) * Decimal("100")
    eff.growth_per_turn = _conversation_growth(turn_context)
    return eff


def _conversation_growth(samples: list[tuple[int, int]]) -> Decimal:
    """Least-squares slope of prompt tokens against conversation turn.

    A steep positive slope means history is being resent verbatim every turn —
    cost then grows quadratically in turn count, which is the single most
    common cause of "our chatbot got expensive as users engaged more". The fix
    (rolling summarisation) is only sellable once you can show the slope.
    """
    if len(samples) < 3:
        return ZERO
    n = Decimal(len(samples))
    sx = sum((Decimal(t) for t, _ in samples), ZERO)
    sy = sum((Decimal(v) for _, v in samples), ZERO)
    sxy = sum((Decimal(t) * Decimal(v) for t, v in samples), ZERO)
    sxx = sum((Decimal(t) * Decimal(t) for t, _ in samples), ZERO)
    denominator = n * sxx - sx * sx
    if denominator == 0:
        return ZERO
    return (n * sxy - sx * sy) / denominator


def token_heatmap(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown] | list[CostBreakdown],
    *,
    row_dimension: str = "team",
    column_dimension: str = "model",
) -> dict[str, dict[str, Decimal]]:
    """Two-dimensional cost matrix backing the dashboard heatmap.

    Returned as a nested mapping rather than a dense matrix because the space
    is extremely sparse — most teams touch three or four models out of thirty —
    and shipping the zeros would dominate the payload.
    """
    cost_by_id = _index_costs(costs)
    row_fn = DIMENSION_EXTRACTORS.get(row_dimension, lambda _e: None)
    col_fn = DIMENSION_EXTRACTORS.get(column_dimension, lambda _e: None)
    matrix: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(lambda: ZERO))
    for event in events:
        cost = cost_by_id.get(str(event.id))
        if cost is None:
            continue
        row = row_fn(event) or "unattributed"
        col = col_fn(event) or "unknown"
        matrix[row][col] += cost.total
    return {r: dict(c) for r, c in matrix.items()}


def cost_flow(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown] | list[CostBreakdown],
    *,
    stages: tuple[str, ...] = ("department", "team", "provider", "model"),
) -> list[dict[str, object]]:
    """Sankey links describing how spend flows through the org and out to
    providers. Emitted as (source, target, value) triples the frontend renders
    directly, so chart config stays declarative."""
    cost_by_id = _index_costs(costs)
    links: dict[tuple[str, str], Decimal] = defaultdict(lambda: ZERO)
    for event in events:
        cost = cost_by_id.get(str(event.id))
        if cost is None or cost.total == ZERO:
            continue
        path = [
            f"{stage}:{(DIMENSION_EXTRACTORS.get(stage, lambda _e: None)(event) or 'unattributed')}"
            for stage in stages
        ]
        for source, target in pairwise(path):
            links[(source, target)] += cost.total
    return [
        {"source": s, "target": t, "value": float(v)}
        for (s, t), v in sorted(links.items(), key=lambda kv: kv[1], reverse=True)
    ]


def _index_costs(
    costs: dict[str, CostBreakdown] | list[CostBreakdown],
) -> dict[str, CostBreakdown]:
    if isinstance(costs, dict):
        return costs
    return {str(c.event_id): c for c in costs}


def top_waste_sources(
    report: TokenAnalyticsReport, *, dimension: str = "feature", limit: int = 10
) -> list[dict[str, object]]:
    """Rank slices by wasted spend — the recommendation engine's entry point."""
    buckets = report.slices.get(dimension, [])
    ranked = sorted(buckets, key=lambda s: s.wasted_cost, reverse=True)[:limit]
    return [
        {
            "key": b.key,
            "wasted_cost": float(b.wasted_cost),
            "total_cost": float(b.cost),
            "waste_ratio": float(safe_div(b.wasted_cost, b.cost) * Decimal("100")),
            "requests": b.requests,
        }
        for b in ranked
        if b.wasted_cost > ZERO
    ]


def compression_savings(
    original_tokens: int, compressed_tokens: int, rate_per_token: Decimal | float | str
) -> Decimal:
    """Dollar value of a prompt-compression opportunity."""
    saved = max(0, original_tokens - compressed_tokens)
    return Decimal(saved) * to_decimal(rate_per_token)
