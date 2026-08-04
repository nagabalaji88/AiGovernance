"""Cost resolution: usage event -> priced, attributed dollars.

This is the most correctness-critical module in the platform. Everything else
is analysis on top of its output, so an error here is silently inherited by
forecasts, chargeback and every dashboard. It is consequently written as pure
functions over immutable inputs with no I/O, which makes it exhaustively
unit-testable and safe to run inside the hot ingestion loop.

## Alternatives considered

- **Price at query time** (store tokens only, join to rate cards on read).
  Pro: re-pricing is free, storage is smaller. Con: every dashboard query
  becomes a temporal join over a slowly-changing dimension, which at our
  cardinality (10^9 rows/quarter) is a non-starter for a 200ms p95 tile.
- **Price at write time only** (what a naive implementation does). Fast reads,
  but a rate-card correction can never be applied to history.
- **Chosen: price at write time, retain inputs for deterministic re-pricing.**
  We persist the resolved cost *and* the rate-card version that produced it.
  Dashboards read pre-computed costs (fast), and a correction is a bounded
  backfill job that re-runs this same pure function over the affected window.
  Storage overhead of the extra columns is ~40 bytes/event — trivial next to
  the query cost it removes.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from app.domain.enums import SELF_HOSTED_PROVIDERS, Provider, TokenClass
from app.domain.money import ZERO, quantize_cost
from app.domain.pricing import (
    InfrastructureCostModel,
    PricingCatalog,
    RateCard,
    default_catalog,
)
from app.domain.usage import CostBreakdown, TokenUsage, UsageEvent


class UnknownModelError(LookupError):
    """Raised when no rate card covers a (provider, model) at a given time.

    Deliberately *not* swallowed into a zero cost. Silently pricing an unknown
    model at $0 is how a platform under-reports spend on exactly the newest,
    most expensive model a team just adopted. The ingestion path catches this,
    prices the event as `unpriced`, and raises an operational alert so the
    catalog gets updated — visible failure beats invisible wrongness.
    """


class CostEngine:
    """Prices usage events against an effective-dated catalog."""

    def __init__(
        self,
        catalog: Optional[PricingCatalog] = None,
        infra_model: Optional[InfrastructureCostModel] = None,
    ) -> None:
        self.catalog = catalog or default_catalog
        self.infra_model = infra_model or InfrastructureCostModel()

    # -- public API ---------------------------------------------------------

    def price(self, event: UsageEvent, *, strict: bool = True) -> CostBreakdown:
        """Resolve the full cost of one event.

        `strict=False` returns a zero-cost breakdown for unknown models instead
        of raising, which the batch backfill uses to make progress over a
        window containing a handful of unrecognised models rather than aborting
        the whole job.
        """
        card = self.catalog.resolve(event.provider, event.model, event.occurred_at)
        if card is None:
            if strict:
                raise UnknownModelError(
                    f"no rate card for {event.provider}/{event.model} "
                    f"effective at {event.occurred_at.isoformat()}"
                )
            return CostBreakdown(event_id=event.id, rate_card_version="unpriced")

        breakdown = CostBreakdown(
            event_id=event.id,
            currency=card.currency,
            rate_card_version=_card_version(card),
        )

        # A non-billable failure (e.g. a 429 rejected before any generation)
        # consumed no provider capacity. We still record the event — the
        # retry-storm detector needs it — but at zero cost.
        if not event.is_billable:
            breakdown.uncached_equivalent = ZERO
            return breakdown

        breakdown.by_token_class = self._token_costs(event.tokens, card)
        breakdown.request_fee = card.per_request

        if event.provider in SELF_HOSTED_PROVIDERS and event.trace.gpu_seconds > ZERO:
            compute, energy, network = self.infra_model.compute_cost(
                gpu_seconds=event.trace.gpu_seconds,
                network_gb=event.trace.network_gb,
            )
            breakdown.compute_cost = quantize_cost(compute)
            breakdown.energy_cost = quantize_cost(energy)
            breakdown.network_cost = quantize_cost(network)

        breakdown.uncached_equivalent = self._uncached_equivalent(event, card, breakdown)
        return breakdown

    def price_many(self, events: list[UsageEvent], *, strict: bool = False) -> list[CostBreakdown]:
        return [self.price(e, strict=strict) for e in events]

    def estimate(
        self,
        *,
        provider: Provider,
        model: str,
        tokens: TokenUsage,
        at: Optional[datetime] = None,
    ) -> Decimal:
        """Cost of a hypothetical call. Backs the simulation engine and the
        pre-flight budget check, where no real event exists yet."""
        card = self.catalog.resolve(provider, model, at)
        if card is None:
            raise UnknownModelError(f"no rate card for {provider}/{model}")
        return quantize_cost(sum(self._token_costs(tokens, card).values(), ZERO) + card.per_request)

    # -- internals ----------------------------------------------------------

    def _token_costs(self, tokens: TokenUsage, card: RateCard) -> dict[TokenClass, Decimal]:
        """Per-class cost, omitting classes that cost nothing.

        Omission (rather than storing explicit zeros) keeps the persisted JSONB
        small — at 10^9 rows, nine always-present keys per row is tens of GB of
        pure padding.
        """
        costs: dict[TokenClass, Decimal] = {}
        for token_class, count in tokens.as_map().items():
            if count <= 0:
                continue
            rate = card.rate_for(token_class)
            if rate == ZERO:
                continue
            costs[token_class] = quantize_cost(Decimal(count) * rate)
        return costs

    def _uncached_equivalent(self, event: UsageEvent, card: RateCard, actual: CostBreakdown) -> Decimal:
        """What this call would have cost with every cache disabled.

        Two savings sources are folded together here:

        1. Provider prompt cache — `cached_input` tokens re-priced at the full
           input rate.
        2. Platform semantic cache — a served-from-cache response did not hit
           the provider at all, so its counterfactual is the full call. The SDK
           reports the token counts it *would* have sent, letting us price the
           avoided call exactly rather than estimating it.
        """
        full_input_rate = card.rate_for(TokenClass.INPUT)

        if event.served_from_semantic_cache:
            # The whole call was avoided; the counterfactual is input + output
            # at full rates, and the actual cost is ~0 (embedding lookup only).
            avoided = Decimal(event.tokens.input) * full_input_rate + Decimal(
                event.tokens.output
            ) * card.rate_for(TokenClass.OUTPUT)
            return quantize_cost(avoided + card.per_request)

        if event.tokens.cached_input <= 0:
            return actual.total

        cached_at_full = Decimal(event.tokens.cached_input) * full_input_rate
        cached_at_discount = actual.by_token_class.get(TokenClass.CACHED_INPUT, ZERO)
        return quantize_cost(actual.total - cached_at_discount + cached_at_full)


def _card_version(card: RateCard) -> str:
    """Stable identifier for the exact prices used.

    Embedded in every cost row so an auditor can answer "which price list
    produced this figure" without reconstructing catalog state from timestamps.
    """
    return f"{card.provider}:{card.model}@{card.effective_from.date().isoformat()}"


def summarise_costs(breakdowns: list[CostBreakdown]) -> dict[str, Decimal]:
    """Totals across a set of priced events, for report headers and tiles."""
    total = sum((b.total for b in breakdowns), ZERO)
    tokens = sum((b.token_cost for b in breakdowns), ZERO)
    infra = sum((b.infrastructure_cost for b in breakdowns), ZERO)
    savings = sum((b.cache_savings for b in breakdowns), ZERO)
    return {
        "total_cost": quantize_cost(total),
        "token_cost": quantize_cost(tokens),
        "infrastructure_cost": quantize_cost(infra),
        "cache_savings": quantize_cost(savings),
        "request_fees": quantize_cost(sum((b.request_fee for b in breakdowns), ZERO)),
    }
