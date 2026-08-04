"""Model routing engine.

Picks the cheapest model that will still do the job, subject to hard
constraints (context size, modality, compliance) and a soft objective
(cost / latency / quality / balanced).

## Why a scored linear utility rather than a learned policy

A bandit or RL policy would, in principle, learn the cost/quality frontier per
task type. Rejected for v1 on three grounds:

1. **Cold start.** A new tenant has no reward signal, so the policy explores —
   which means deliberately routing production traffic to bad models to learn.
   That is an unacceptable first-week experience.
2. **Explainability.** "Why did my request go to a cheaper model?" must have an
   answer a developer can read. A weight vector over four normalised features
   produces exactly that; a learned policy produces a shrug.
3. **Safety.** Compliance constraints (data residency, no-train guarantees,
   approved-vendor lists) are hard boolean filters, not soft penalties. A
   scoring model that can trade away a compliance constraint for cost is a
   governance incident waiting to happen — so those are enforced as filters
   *before* scoring, and cannot be outvoted by any weight.

Measured quality feeds back into the scorer through `quality_overrides`, which
is where the learning actually happens: the *inputs* improve continuously from
observed task success, while the decision function stays inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from app.domain.enums import (
    ModelType,
    Provider,
    RoutingObjective,
    TaskComplexity,
    TokenClass,
)
from app.domain.money import ZERO, quantize_cost, safe_div, to_decimal
from app.domain.pricing import PricingCatalog, RateCard, default_catalog
from app.domain.usage import TokenUsage

#: Objective -> (cost weight, latency weight, quality weight, reliability weight).
#: Weights sum to 1 within each row so scores are comparable across objectives.
OBJECTIVE_WEIGHTS: dict[RoutingObjective, tuple[Decimal, Decimal, Decimal, Decimal]] = {
    RoutingObjective.CHEAPEST: (Decimal("0.75"), Decimal("0.05"), Decimal("0.15"), Decimal("0.05")),
    RoutingObjective.FASTEST: (Decimal("0.10"), Decimal("0.70"), Decimal("0.15"), Decimal("0.05")),
    RoutingObjective.HIGHEST_QUALITY: (Decimal("0.05"), Decimal("0.05"), Decimal("0.85"), Decimal("0.05")),
    RoutingObjective.BALANCED: (Decimal("0.35"), Decimal("0.20"), Decimal("0.35"), Decimal("0.10")),
    RoutingObjective.REASONING: (Decimal("0.15"), Decimal("0.05"), Decimal("0.75"), Decimal("0.05")),
    RoutingObjective.VISION: (Decimal("0.30"), Decimal("0.15"), Decimal("0.50"), Decimal("0.05")),
    RoutingObjective.EMBEDDING: (Decimal("0.60"), Decimal("0.20"), Decimal("0.15"), Decimal("0.05")),
    RoutingObjective.LOCAL_ONLY: (Decimal("0.50"), Decimal("0.25"), Decimal("0.20"), Decimal("0.05")),
}

#: Minimum quality index a model must clear to be considered for a given task
#: complexity. This is the guard that stops "cheapest" from routing an expert
#: legal-analysis task to an 8B model and calling it a saving.
COMPLEXITY_QUALITY_FLOOR: dict[TaskComplexity, Decimal] = {
    TaskComplexity.TRIVIAL: Decimal("0.60"),
    TaskComplexity.SIMPLE: Decimal("0.72"),
    TaskComplexity.MODERATE: Decimal("0.82"),
    TaskComplexity.COMPLEX: Decimal("0.90"),
    TaskComplexity.EXPERT: Decimal("0.94"),
}


@dataclass(slots=True)
class RoutingConstraints:
    """Hard requirements. Any candidate failing one is removed, not penalised."""

    min_context_tokens: int = 0
    requires_vision: bool = False
    requires_tools: bool = False
    requires_reasoning: bool = False
    max_latency_ms: int | None = None
    max_cost_per_request: Decimal | None = None
    #: Vendor allow-list from the tenant's procurement/compliance policy.
    allowed_providers: set[Provider] | None = None
    blocked_providers: set[Provider] = field(default_factory=set)
    #: Data residency / no-training requirements force self-hosted or an
    #: approved regional deployment.
    require_self_hosted: bool = False
    allowed_model_types: set[ModelType] | None = None


@dataclass(slots=True)
class RoutingRequest:
    expected_input_tokens: int
    expected_output_tokens: int
    objective: RoutingObjective = RoutingObjective.BALANCED
    complexity: TaskComplexity = TaskComplexity.MODERATE
    constraints: RoutingConstraints = field(default_factory=RoutingConstraints)
    #: Measured quality per "provider/model", overriding the catalog prior.
    quality_overrides: dict[str, Decimal] = field(default_factory=dict)
    #: Observed success rate per "provider/model" from the reliability tracker.
    reliability: dict[str, Decimal] = field(default_factory=dict)
    expected_cached_input_tokens: int = 0

    @property
    def token_estimate(self) -> TokenUsage:
        return TokenUsage(
            input=self.expected_input_tokens,
            output=self.expected_output_tokens,
            cached_input=self.expected_cached_input_tokens,
        )

    @property
    def required_context(self) -> int:
        needed = self.expected_input_tokens + self.expected_output_tokens
        return max(needed, self.constraints.min_context_tokens)


@dataclass(slots=True)
class RoutingCandidate:
    provider: Provider
    model: str
    estimated_cost: Decimal
    estimated_latency_ms: int
    quality: Decimal
    reliability: Decimal
    score: Decimal = ZERO
    #: Per-factor contributions, returned to the caller so a routing decision
    #: can be explained without re-deriving it.
    factors: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(slots=True)
class RoutingDecision:
    selected: RoutingCandidate | None
    alternatives: list[RoutingCandidate] = field(default_factory=list)
    baseline: RoutingCandidate | None = None
    rejected: dict[str, str] = field(default_factory=dict)
    rationale: str = ""

    @property
    def savings_vs_baseline(self) -> Decimal:
        if not self.selected or not self.baseline:
            return ZERO
        delta = self.baseline.estimated_cost - self.selected.estimated_cost
        return delta if delta > ZERO else ZERO

    @property
    def savings_pct(self) -> Decimal:
        if not self.baseline or self.baseline.estimated_cost == ZERO:
            return ZERO
        return safe_div(self.savings_vs_baseline, self.baseline.estimated_cost) * Decimal("100")

    @property
    def quality_delta(self) -> Decimal:
        if not self.selected or not self.baseline:
            return ZERO
        return self.selected.quality - self.baseline.quality


class ModelRouter:
    def __init__(self, catalog: PricingCatalog | None = None) -> None:
        self.catalog = catalog or default_catalog

    def route(self, request: RoutingRequest, *, baseline_model: str | None = None) -> RoutingDecision:
        """Select a model and explain the choice."""
        rejected: dict[str, str] = {}
        candidates: list[RoutingCandidate] = []

        for card in self.catalog.all_cards():
            reason = self._reject_reason(card, request)
            if reason:
                rejected[f"{card.provider}/{card.model}"] = reason
                continue
            candidates.append(self._build_candidate(card, request))

        if not candidates:
            return RoutingDecision(
                selected=None,
                rejected=rejected,
                rationale=(
                    "No model satisfies the hard constraints. Most commonly this means the "
                    "required context window exceeds every approved model, or the compliance "
                    "allow-list excludes all candidates with the needed modality."
                ),
            )

        self._score(candidates, request)
        candidates.sort(key=lambda c: c.score, reverse=True)
        selected = candidates[0]

        baseline = None
        if baseline_model:
            baseline = next((c for c in candidates if c.model == baseline_model), None)
            if baseline is None:
                baseline = self._baseline_from_catalog(baseline_model, request)

        return RoutingDecision(
            selected=selected,
            alternatives=candidates[1:6],
            baseline=baseline,
            rejected=rejected,
            rationale=self._explain(selected, request, baseline),
        )

    # -- internals ----------------------------------------------------------

    def _reject_reason(self, card: RateCard, request: RoutingRequest) -> str | None:
        c = request.constraints
        if card.context_window < request.required_context:
            return f"context window {card.context_window} < required {request.required_context}"
        if c.requires_vision and not card.supports_vision:
            return "no vision support"
        if c.requires_tools and not card.supports_tools:
            return "no tool-calling support"
        if c.requires_reasoning and card.model_type is not ModelType.REASONING:
            return "not a reasoning model"
        if c.allowed_providers and card.provider not in c.allowed_providers:
            return "provider not on the approved list"
        if card.provider in c.blocked_providers:
            return "provider explicitly blocked by policy"
        if c.allowed_model_types and card.model_type not in c.allowed_model_types:
            return "model type not permitted for this workload"
        if c.require_self_hosted and card.model_type is not ModelType.LOCAL:
            return "data residency policy requires a self-hosted model"
        if card.max_output_tokens < request.expected_output_tokens:
            return f"max output {card.max_output_tokens} < required {request.expected_output_tokens}"

        quality = self._quality_of(card, request)
        floor = COMPLEXITY_QUALITY_FLOOR[request.complexity]
        if quality < floor:
            return f"quality {quality} below the {floor} floor for {request.complexity} tasks"

        latency = self._latency_of(card, request)
        if c.max_latency_ms is not None and latency > c.max_latency_ms:
            return f"estimated latency {latency}ms exceeds the {c.max_latency_ms}ms budget"

        cost = self._cost_of(card, request)
        if c.max_cost_per_request is not None and cost > c.max_cost_per_request:
            return f"estimated cost {cost} exceeds the per-request ceiling"
        return None

    def _quality_of(self, card: RateCard, request: RoutingRequest) -> Decimal:
        return request.quality_overrides.get(f"{card.provider}/{card.model}", card.quality_index)

    def _latency_of(self, card: RateCard, request: RoutingRequest) -> int:
        """Latency is dominated by output token count, not input.

        Prefill is parallel and fast; decode is sequential and slow. Modelling
        latency as proportional to output tokens with a fixed overhead matches
        observed behaviour far better than a flat per-model constant, and it is
        why a "fast" model can still be the wrong choice for a long generation.
        """
        decode = int(card.latency_ms_per_1k_output * request.expected_output_tokens / 1000)
        prefill = int(request.expected_input_tokens / 1000) * 12
        return decode + prefill + 120

    def _cost_of(self, card: RateCard, request: RoutingRequest) -> Decimal:
        tokens = request.token_estimate
        total = ZERO
        for token_class, count in tokens.as_map().items():
            if count > 0:
                total += Decimal(count) * card.rate_for(token_class)
        return quantize_cost(total + card.per_request)

    def _build_candidate(self, card: RateCard, request: RoutingRequest) -> RoutingCandidate:
        key = f"{card.provider}/{card.model}"
        return RoutingCandidate(
            provider=card.provider,
            model=card.model,
            estimated_cost=self._cost_of(card, request),
            estimated_latency_ms=self._latency_of(card, request),
            quality=self._quality_of(card, request),
            reliability=request.reliability.get(key, Decimal("0.99")),
        )

    def _score(self, candidates: list[RoutingCandidate], request: RoutingRequest) -> None:
        """Normalise each factor to 0-1 across the candidate set, then combine.

        Min-max normalisation *within the candidate set* rather than against
        absolute scales, because the meaningful question is "how does this
        option compare to the others available for this request", not "how does
        it compare to all models ever". A set of uniformly expensive candidates
        should still produce a clear cheapest choice.
        """
        w_cost, w_latency, w_quality, w_reliability = OBJECTIVE_WEIGHTS[request.objective]

        costs = [c.estimated_cost for c in candidates]
        latencies = [Decimal(c.estimated_latency_ms) for c in candidates]
        qualities = [c.quality for c in candidates]

        def normalise(value: Decimal, values: list[Decimal], *, invert: bool) -> Decimal:
            lo, hi = min(values), max(values)
            if hi == lo:
                return Decimal("1")
            scaled = safe_div(value - lo, hi - lo)
            return Decimal("1") - scaled if invert else scaled

        for candidate in candidates:
            # Cost and latency are costs to minimise, so they are inverted:
            # cheapest/fastest scores 1.
            cost_score = normalise(candidate.estimated_cost, costs, invert=True)
            latency_score = normalise(Decimal(candidate.estimated_latency_ms), latencies, invert=True)
            quality_score = normalise(candidate.quality, qualities, invert=False)
            reliability_score = candidate.reliability

            candidate.score = (
                cost_score * w_cost
                + latency_score * w_latency
                + quality_score * w_quality
                + reliability_score * w_reliability
            )
            candidate.factors = {
                "cost": float(cost_score),
                "latency": float(latency_score),
                "quality": float(quality_score),
                "reliability": float(reliability_score),
            }

    def _baseline_from_catalog(self, baseline_model: str, request: RoutingRequest) -> RoutingCandidate | None:
        for card in self.catalog.all_cards():
            if card.model == baseline_model:
                return self._build_candidate(card, request)
        return None

    def _explain(
        self,
        selected: RoutingCandidate,
        request: RoutingRequest,
        baseline: RoutingCandidate | None,
    ) -> str:
        parts = [
            f"Selected {selected.key} for a {request.complexity} task under the "
            f"{request.objective} objective."
        ]
        parts.append(
            f"Estimated {quantize_cost(selected.estimated_cost)} per request at "
            f"~{selected.estimated_latency_ms}ms and quality index {selected.quality}."
        )
        if baseline and baseline.key != selected.key:
            delta = baseline.estimated_cost - selected.estimated_cost
            if delta > ZERO:
                pct = safe_div(delta, baseline.estimated_cost) * Decimal("100")
                quality_note = (
                    "at equal or better measured quality"
                    if selected.quality >= baseline.quality
                    else f"with a {baseline.quality - selected.quality} quality index reduction"
                )
                parts.append(f"That is {pct:.1f}% cheaper than {baseline.key} ({quality_note}).")
            else:
                parts.append(
                    f"This costs more than {baseline.key}, justified by the {request.objective} objective."
                )
        return " ".join(parts)


def classify_complexity(
    *,
    input_tokens: int,
    requires_reasoning: bool = False,
    requires_tools: bool = False,
    has_structured_output: bool = False,
    domain_sensitive: bool = False,
) -> TaskComplexity:
    """Heuristic task-complexity classifier.

    Rule-based on purpose. An LLM classifier would be more nuanced but adds a
    model call to the front of every routed request — latency and cost on the
    critical path of a *cost optimization* product, which is the wrong trade.
    Callers who know their task's complexity should pass it explicitly; this is
    the fallback for un-annotated traffic.
    """
    if domain_sensitive:
        return TaskComplexity.EXPERT
    score = 0
    if input_tokens > 32_000:
        score += 2
    elif input_tokens > 8_000:
        score += 1
    if requires_reasoning:
        score += 2
    if requires_tools:
        score += 1
    if has_structured_output:
        score += 1

    if score >= 5:
        return TaskComplexity.EXPERT
    if score >= 3:
        return TaskComplexity.COMPLEX
    if score >= 2:
        return TaskComplexity.MODERATE
    if score >= 1:
        return TaskComplexity.SIMPLE
    return TaskComplexity.TRIVIAL


def compare_models(
    catalog: PricingCatalog,
    models: list[tuple[Provider, str]],
    *,
    input_tokens: int,
    output_tokens: int,
    monthly_requests: int,
) -> list[dict[str, object]]:
    """Side-by-side cost projection, backing the model comparison dashboard."""
    rows: list[dict[str, object]] = []
    for provider, model in models:
        card = catalog.resolve(provider, model)
        if card is None:
            continue
        per_request = (
            Decimal(input_tokens) * card.rate_for(TokenClass.INPUT)
            + Decimal(output_tokens) * card.rate_for(TokenClass.OUTPUT)
            + card.per_request
        )
        monthly = quantize_cost(per_request * Decimal(monthly_requests))
        rows.append(
            {
                "provider": str(provider),
                "model": model,
                "cost_per_request": float(quantize_cost(per_request)),
                "monthly_cost": float(monthly),
                "annual_cost": float(quantize_cost(monthly * Decimal("12"))),
                "quality_index": float(card.quality_index),
                "context_window": card.context_window,
                "estimated_latency_ms": int(card.latency_ms_per_1k_output * output_tokens / 1000),
                "supports_prompt_cache": card.supports_prompt_cache,
                "cost_per_quality_point": float(safe_div(per_request, to_decimal(card.quality_index))),
            }
        )
    rows.sort(key=lambda r: r["monthly_cost"])  # type: ignore[arg-type,return-value]
    return rows
