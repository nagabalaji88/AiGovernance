"""Cache strategy advisor.

Answers three questions per workload: *which* cache tier applies, what TTL to
set, and what it is worth in dollars.

## The four tiers, and why they are not interchangeable

- **Provider prompt cache** — keyed on the exact static prefix. Saves 75-95%
  of the prefix's input cost. No risk: the provider guarantees identical
  semantics.
- **Exact response cache** — keyed on a hash of the full request. Saves 100%
  of the call. Only risk is staleness.
- **Semantic cache** — keyed on embedding similarity. Saves 100% of the call,
  but can return **wrong answers** if the threshold is loose.
- **Embedding cache** — keyed on a hash of the source text. Saves 100% of
  re-embedding. No risk: embeddings are deterministic per model.

The semantic cache is the only one that can return a *wrong* answer, because
"similar enough" is a judgement call. A threshold of 0.95 cosine similarity
sounds conservative until you notice that "what is our refund policy for EU
customers" and "what is our refund policy for US customers" sit around 0.97.
This module therefore defaults the semantic threshold high (0.97), requires an
explicit opt-in per prompt family, and recommends it only where the observed
query distribution is genuinely repetitive. Saving money by confidently
answering the wrong question is not a saving.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from statistics import median
from typing import Union

from app.domain.money import ZERO, quantize_cost, safe_div
from app.domain.usage import CostBreakdown, UsageEvent

#: Cosine similarity floor for semantic cache hits. High by design; see module
#: docstring. Tenants may lower it per prompt family after evaluating recall.
DEFAULT_SEMANTIC_THRESHOLD = Decimal("0.97")

#: A prefix must be at least this large before prompt caching pays for itself —
#: providers charge a premium to *write* a cache entry, so caching a 200-token
#: preamble that is read twice is a net loss.
MIN_CACHEABLE_PREFIX_TOKENS = 1_024


@dataclass
class CacheRecommendation:
    tier: str
    scope_key: str
    rationale: str
    estimated_monthly_savings: Decimal
    expected_hit_rate: Decimal
    recommended_ttl_seconds: int
    confidence: Decimal
    risk: str = "low"
    implementation_note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "scope_key": self.scope_key,
            "rationale": self.rationale,
            "estimated_monthly_savings": float(self.estimated_monthly_savings),
            "expected_hit_rate": float(self.expected_hit_rate),
            "recommended_ttl_seconds": self.recommended_ttl_seconds,
            "confidence": float(self.confidence),
            "risk": self.risk,
            "implementation_note": self.implementation_note,
        }


@dataclass
class CachePerformance:
    """Realised performance of caching already in place."""

    provider_cache_hit_ratio: Decimal = ZERO
    semantic_cache_hit_ratio: Decimal = ZERO
    realised_savings: Decimal = ZERO
    #: Spend on requests that *could* have hit a cache but did not.
    missed_savings: Decimal = ZERO
    total_cost: Decimal = ZERO

    @property
    def capture_rate(self) -> Decimal:
        """Share of the total caching opportunity actually being captured."""
        opportunity = self.realised_savings + self.missed_savings
        return safe_div(self.realised_savings, opportunity) * Decimal("100")


def _window_days(events: list[UsageEvent]) -> Decimal:
    if len(events) < 2:
        return Decimal("1")
    ordered = sorted(e.occurred_at for e in events)
    span = (ordered[-1] - ordered[0]).total_seconds() / 86_400
    return Decimal(str(max(span, 0.041)))  # floor at ~1 hour to avoid absurd scaling


def _to_monthly(amount: Decimal, window_days: Decimal) -> Decimal:
    """Scale an observed-window figure to a monthly rate.

    Extrapolating a 2-hour sample to a month is statistically indefensible, so
    callers surface `confidence` alongside, which we degrade for short windows.
    """
    return quantize_cost(amount * safe_div(Decimal("30"), window_days))


def _confidence_for_window(window_days: Decimal, sample_size: int) -> Decimal:
    """Confidence degrades with short windows and small samples."""
    window_factor = min(Decimal("1"), safe_div(window_days, Decimal("7")))
    sample_factor = min(Decimal("1"), safe_div(Decimal(sample_size), Decimal("500")))
    return (window_factor * Decimal("0.5") + sample_factor * Decimal("0.5")).quantize(Decimal("0.01"))


def recommend_prompt_cache(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown],
    *,
    discount_ratio: Decimal = Decimal("0.9"),
) -> list[CacheRecommendation]:
    """Recommend provider prompt caching for large, stable static prefixes.

    `discount_ratio` is the share of the input rate saved on a cache read
    (~90% for the major providers). We apply it only to the static regions —
    system prompt, tool definitions, few-shot — because those are what a prefix
    cache can actually cover.
    """
    by_template: dict[str, dict[str, object]] = defaultdict(lambda: {"calls": 0, "static": [], "cost": ZERO})
    for event in events:
        if event.tokens.cached_input > 0:
            continue  # already caching
        key = str(event.prompt_template_id or event.prompt_fingerprint or f"{event.provider}/{event.model}")
        entry = by_template[key]
        entry["calls"] = int(entry["calls"]) + 1  # type: ignore[arg-type]
        static = (
            event.trace.system_prompt_tokens
            + event.trace.tool_definition_tokens
            + event.trace.few_shot_tokens
        )
        entry["static"].append(static)  # type: ignore[union-attr]
        cost = costs.get(str(event.id))
        if cost:
            entry["cost"] = entry["cost"] + cost.total  # type: ignore[operator]

    window = _window_days(events)
    recommendations: list[CacheRecommendation] = []
    for key, entry in by_template.items():
        calls = int(entry["calls"])  # type: ignore[arg-type]
        statics: list[int] = entry["static"]  # type: ignore[assignment]
        if calls < 20 or not statics:
            continue
        typical_static = int(median(statics))
        if typical_static < MIN_CACHEABLE_PREFIX_TOKENS:
            continue

        # Static share of spend, discounted by the cache read rate. The first
        # call in each TTL window still pays full price plus a write premium,
        # which we approximate by discounting the raw saving by 10%.
        spend: Decimal = entry["cost"]  # type: ignore[assignment]
        total_prompt = sum(statics)
        static_share = safe_div(Decimal(typical_static * calls), Decimal(max(total_prompt, 1)))
        raw_saving = spend * min(static_share, Decimal("1")) * discount_ratio * Decimal("0.9")
        monthly = _to_monthly(raw_saving, window)
        if monthly < Decimal("5"):
            continue

        recommendations.append(
            CacheRecommendation(
                tier="provider_prompt_cache",
                scope_key=key,
                rationale=(
                    f"{typical_static:,} tokens of static prefix repeat across {calls} calls "
                    "with no prompt cache in use. Provider prefix caching bills these at "
                    "roughly a tenth of the input rate."
                ),
                estimated_monthly_savings=monthly,
                expected_hit_rate=Decimal("0.85"),
                recommended_ttl_seconds=300,
                confidence=_confidence_for_window(window, calls),
                risk="none",
                implementation_note=(
                    "Move the system prompt, tool definitions and few-shot block to the very "
                    "start of the message array and mark the cache breakpoint after them. "
                    "Prefix caching requires a byte-identical prefix, so any per-request value "
                    "(timestamp, user id) must move below the breakpoint."
                ),
            )
        )
    return recommendations


def recommend_response_cache(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown],
    *,
    min_repeats: int = 3,
) -> list[CacheRecommendation]:
    """Exact-match response caching for repeated identical requests.

    Opportunities are aggregated **per feature**, not per prompt. A FAQ
    endpoint serving twelve distinct questions a few hundred times each is one
    decision ("cache this endpoint's responses"), not twelve. Emitting one
    recommendation per fingerprint would both bury the real signal under a
    wall of near-identical rows and, worse, drop the whole opportunity when
    each individual prompt falls below the reporting floor while their sum is
    substantial.

    TTL is derived from the observed inter-arrival time of duplicates rather
    than a fixed default: a prompt repeated every few seconds needs a short TTL
    to stay fresh but will still hit constantly, whereas one repeated daily
    needs a long TTL to hit at all. A single global TTL is why most response
    caches under-deliver.
    """
    scopes: dict[str, dict[str, list[UsageEvent]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event.prompt_fingerprint and not event.served_from_semantic_cache:
            scope = event.attribution.feature or event.attribution.application or "unattributed"
            scopes[scope][event.prompt_fingerprint].append(event)

    window = _window_days(events)
    recommendations: list[CacheRecommendation] = []

    for scope, groups in scopes.items():
        repeated = {fp: group for fp, group in groups.items() if len(group) >= min_repeats}
        if not repeated:
            continue

        recoverable = ZERO
        gaps: list[float] = []
        duplicate_calls = 0
        total_calls = 0
        for group in repeated.values():
            spend = sum((costs[str(e.id)].total for e in group if str(e.id) in costs), ZERO)
            # The first call of each distinct prompt is unavoidable; the rest
            # are cacheable.
            recoverable += spend * safe_div(Decimal(len(group) - 1), Decimal(len(group)))
            duplicate_calls += len(group) - 1
            total_calls += len(group)
            ordered = sorted(e.occurred_at for e in group)
            gaps.extend((b - a).total_seconds() for a, b in zip(ordered, ordered[1:]))

        monthly = _to_monthly(recoverable, window)
        if monthly < Decimal("5"):
            continue

        typical_gap = median(gaps) if gaps else 3600.0
        # TTL at ~3x the typical gap captures most repeats while bounding
        # staleness to a window the data owner can reason about.
        ttl = int(min(max(typical_gap * 3, 60), 86_400))
        top = sorted(repeated.items(), key=lambda kv: len(kv[1]), reverse=True)[:3]

        recommendations.append(
            CacheRecommendation(
                tier="exact_response_cache",
                scope_key=scope,
                rationale=(
                    f"{len(repeated)} distinct prompts in '{scope}' were re-sent to the provider "
                    f"{duplicate_calls:,} times beyond their first call "
                    f"(typically every {typical_gap / 60:.1f} minutes). "
                    f"Most repeated: {', '.join(f'{fp} x{len(g)}' for fp, g in top)}."
                ),
                estimated_monthly_savings=monthly,
                expected_hit_rate=safe_div(Decimal(duplicate_calls), Decimal(total_calls or 1)),
                recommended_ttl_seconds=ttl,
                confidence=_confidence_for_window(window, total_calls),
                risk="low",
                implementation_note=(
                    "Key on a hash of (model, full message array, temperature, tool schema). "
                    "Omitting temperature or the tool schema from the key is the usual cause of "
                    "surprising cache hits. Bypass the cache when the caller sets temperature > 0 "
                    "and genuinely wants variation."
                ),
            )
        )
    return recommendations


def recommend_semantic_cache(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown],
    *,
    threshold: Decimal = DEFAULT_SEMANTIC_THRESHOLD,
) -> list[CacheRecommendation]:
    """Semantic caching, recommended conservatively and only where warranted.

    The gate is the *exact repeat rate*. If users are already re-issuing
    byte-identical queries, the underlying intent distribution is repetitive
    and near-duplicate phrasings of those same intents are near-certain — so
    semantic matching will capture traffic that exact matching misses. If
    almost every request is unique, hit rates will be low and the embedding
    lookup cost per miss starts to erode the saving.

    Savings are estimated only against the traffic exact caching would *not*
    already cover, so this recommendation can be stacked with the exact
    response cache without double-counting the same dollars.
    """
    by_feature: dict[str, list[UsageEvent]] = defaultdict(list)
    for event in events:
        key = event.attribution.feature or event.attribution.application or "unattributed"
        by_feature[key].append(event)

    window = _window_days(events)
    recommendations: list[CacheRecommendation] = []
    for feature, group in by_feature.items():
        if len(group) < 100:
            continue
        fingerprints = Counter(e.prompt_fingerprint for e in group if e.prompt_fingerprint)
        if not fingerprints:
            continue
        distinct = len(fingerprints)
        total = sum(fingerprints.values())
        duplicates = total - distinct
        repeat_rate = safe_div(Decimal(duplicates), Decimal(total))

        # Too few distinct prompts means exact caching already solves it and a
        # semantic layer adds risk for no incremental gain.
        if distinct < 20 or repeat_rate < Decimal("0.15"):
            continue

        spend = sum((costs[str(e.id)].total for e in group if str(e.id) in costs), ZERO)
        # Only the traffic exact matching leaves on the table is addressable
        # here. Of that residual, assume semantic matching converts a fraction
        # proportional to how repetitive the workload already is.
        residual_share = safe_div(Decimal(distinct), Decimal(total))
        expected_hit = residual_share * repeat_rate * Decimal("0.5")
        monthly = _to_monthly(spend * expected_hit, window)
        if monthly < Decimal("25"):
            continue

        recommendations.append(
            CacheRecommendation(
                tier="semantic_cache",
                scope_key=feature,
                rationale=(
                    f"'{feature}' shows {repeat_rate * 100:.0f}% byte-identical repeats across "
                    f"{distinct:,} distinct prompts, so users are asking the same questions in "
                    "varying phrasings. Semantic matching captures the near-duplicates that "
                    "exact caching misses."
                ),
                estimated_monthly_savings=monthly,
                expected_hit_rate=expected_hit,
                recommended_ttl_seconds=3_600,
                confidence=_confidence_for_window(window, len(group)) * Decimal("0.7"),
                risk="medium",
                implementation_note=(
                    f"Start at a {threshold} cosine threshold and shadow-run it: log what "
                    "would have been served from cache alongside the live answer, and only "
                    "enable serving once a human has reviewed a sample. Never enable semantic "
                    "caching on prompts whose answer depends on entities that vary between "
                    "similar phrasings (customer id, region, date), because those are exactly "
                    "the queries that embed closest together."
                ),
            )
        )
    return recommendations


def recommend_embedding_cache(
    events: list[UsageEvent], costs: dict[str, CostBreakdown]
) -> list[CacheRecommendation]:
    """Embeddings are deterministic — re-embedding unchanged text is pure waste."""
    embedding_events = [e for e in events if e.tokens.embedding > 0]
    if len(embedding_events) < 50:
        return []
    fingerprints = Counter(e.prompt_fingerprint for e in embedding_events if e.prompt_fingerprint)
    repeats = sum(c - 1 for c in fingerprints.values() if c > 1)
    if repeats < 10:
        return []

    spend = sum((costs[str(e.id)].total for e in embedding_events if str(e.id) in costs), ZERO)
    total = sum(fingerprints.values())
    recoverable = spend * safe_div(Decimal(repeats), Decimal(max(total, 1)))
    window = _window_days(embedding_events)
    monthly = _to_monthly(recoverable, window)
    if monthly < Decimal("2"):
        return []

    return [
        CacheRecommendation(
            tier="embedding_cache",
            scope_key="embeddings",
            rationale=(
                f"{repeats} of {total} embedding calls re-embedded text that had already been "
                "embedded. Embeddings are deterministic per model, so a hit is always correct."
            ),
            estimated_monthly_savings=monthly,
            expected_hit_rate=safe_div(Decimal(repeats), Decimal(max(total, 1))),
            recommended_ttl_seconds=2_592_000,  # 30 days
            confidence=Decimal("0.9"),
            risk="none",
            implementation_note=(
                "Key on (embedding_model, sha256(normalised_text)). The model must be in the "
                "key — vectors from different models are not interchangeable. Invalidate the "
                "whole namespace on a model upgrade rather than trying to migrate vectors."
            ),
        )
    ]


def measure_performance(events: list[UsageEvent], costs: dict[str, CostBreakdown]) -> CachePerformance:
    """Realised vs. missed caching value over the window."""
    perf = CachePerformance()
    cacheable_spend: Decimal = ZERO
    seen: set[str] = set()
    provider_hits = 0
    provider_eligible = 0
    semantic_hits = 0

    for event in events:
        cost = costs.get(str(event.id))
        amount = cost.total if cost else ZERO
        perf.total_cost += amount
        perf.realised_savings += cost.cache_savings if cost else ZERO

        if event.served_from_semantic_cache:
            semantic_hits += 1
        if event.tokens.input > 0 or event.tokens.cached_input > 0:
            provider_eligible += 1
            if event.tokens.cached_input > 0:
                provider_hits += 1

        fp = event.prompt_fingerprint
        if fp and not event.served_from_semantic_cache:
            if fp in seen:
                cacheable_spend += amount
            seen.add(fp)

    perf.missed_savings = cacheable_spend
    perf.provider_cache_hit_ratio = safe_div(Decimal(provider_hits), Decimal(provider_eligible))
    perf.semantic_cache_hit_ratio = safe_div(Decimal(semantic_hits), Decimal(len(events) or 1))
    return perf


def advise(
    events: list[UsageEvent], costs: Union[dict[str, CostBreakdown], list[CostBreakdown]]
) -> list[CacheRecommendation]:
    """Full cache advisory, ranked by monthly value."""
    cost_map = costs if isinstance(costs, dict) else {str(c.event_id): c for c in costs}
    out: list[CacheRecommendation] = []
    out.extend(recommend_prompt_cache(events, cost_map))
    out.extend(recommend_response_cache(events, cost_map))
    out.extend(recommend_semantic_cache(events, cost_map))
    out.extend(recommend_embedding_cache(events, cost_map))
    out.sort(key=lambda r: r.estimated_monthly_savings, reverse=True)
    return out
