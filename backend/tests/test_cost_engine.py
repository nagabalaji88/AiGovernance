"""Cost engine correctness.

These are the highest-value tests in the suite: every downstream number
inherits an error here, and a silent 10% costing bug would be invisible until
it reached an invoice.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.domain.enums import Provider, RequestStatus, TokenClass
from app.domain.pricing import (
    InfrastructureCostModel,
    PricingCatalog,
    RateCard,
    default_catalog,
)
from app.domain.usage import RequestTrace, TokenUsage, UsageEvent
from app.services.cost_engine import CostEngine, UnknownModelError, summarise_costs


@pytest.fixture
def engine() -> CostEngine:
    return CostEngine()


def make_event(**kwargs) -> UsageEvent:
    defaults = {
        "provider": Provider.OPENAI,
        "model": "gpt-4.1",
        "tokens": TokenUsage(input=1000, output=500),
    }
    defaults.update(kwargs)
    return UsageEvent(**defaults)  # type: ignore[arg-type]


class TestBasicPricing:
    def test_prices_input_and_output_at_distinct_rates(self, engine: CostEngine) -> None:
        # gpt-4.1: $2.00/1M input, $8.00/1M output.
        event = make_event(tokens=TokenUsage(input=1_000_000, output=1_000_000))
        cost = engine.price(event)
        assert cost.by_token_class[TokenClass.INPUT] == Decimal("2.0000000000")
        assert cost.by_token_class[TokenClass.OUTPUT] == Decimal("8.0000000000")
        assert cost.total == Decimal("10.0000000000")

    def test_zero_count_classes_are_omitted_not_stored_as_zero(self, engine: CostEngine) -> None:
        """Keeps the persisted JSONB small at 10^9 rows."""
        cost = engine.price(make_event(tokens=TokenUsage(input=100, output=0)))
        assert TokenClass.OUTPUT not in cost.by_token_class
        assert TokenClass.REASONING not in cost.by_token_class

    def test_unknown_model_raises_rather_than_pricing_at_zero(self, engine: CostEngine) -> None:
        """Silent $0 pricing under-reports exactly the newest, priciest models."""
        with pytest.raises(UnknownModelError):
            engine.price(make_event(model="gpt-99-turbo"))

    def test_non_strict_mode_marks_unpriced_instead_of_raising(self, engine: CostEngine) -> None:
        cost = engine.price(make_event(model="gpt-99-turbo"), strict=False)
        assert cost.total == Decimal("0")
        assert cost.rate_card_version == "unpriced"

    def test_reasoning_tokens_billed_at_output_rate(self, engine: CostEngine) -> None:
        """Under-modelling hidden reasoning tokens causes 3-5x budget overruns."""
        event = make_event(
            provider=Provider.OPENAI,
            model="gpt-5",
            tokens=TokenUsage(input=0, output=0, reasoning=1_000_000),
        )
        cost = engine.price(event)
        assert cost.by_token_class[TokenClass.REASONING] == Decimal("10.0000000000")


class TestBillability:
    def test_rate_limited_request_costs_nothing(self, engine: CostEngine) -> None:
        """A 429 rejected before generation consumed no provider capacity."""
        event = make_event(status=RequestStatus.RATE_LIMITED)
        assert engine.price(event).total == Decimal("0")

    def test_timeout_is_still_billable(self, engine: CostEngine) -> None:
        """Providers bill for whatever was generated before the cut-off."""
        event = make_event(status=RequestStatus.TIMEOUT)
        assert engine.price(event).total > Decimal("0")

    @pytest.mark.parametrize(
        "status,expected",
        [
            (RequestStatus.SUCCESS, False),
            (RequestStatus.ERROR, True),
            (RequestStatus.TIMEOUT, True),
            (RequestStatus.RATE_LIMITED, False),
        ],
    )
    def test_waste_classification(self, status: RequestStatus, expected: bool) -> None:
        assert make_event(status=status).is_wasted is expected

    def test_retry_of_a_parent_request_counts_as_waste(self) -> None:
        event = make_event(trace=RequestTrace(retry_count=2, parent_request_id="req-1"))
        assert event.is_wasted


class TestCaching:
    def test_cached_tokens_are_a_separate_cheaper_class_not_a_discount(self, engine: CostEngine) -> None:
        """Treating cached tokens as a subset of input mis-prices by up to 90%."""
        event = make_event(tokens=TokenUsage(input=0, output=0, cached_input=1_000_000))
        cost = engine.price(event)
        # gpt-4.1 cached input is $0.50/1M vs $2.00/1M full rate.
        assert cost.by_token_class[TokenClass.CACHED_INPUT] == Decimal("0.5000000000")

    def test_uncached_equivalent_quantifies_the_realised_saving(self, engine: CostEngine) -> None:
        event = make_event(tokens=TokenUsage(input=0, output=0, cached_input=1_000_000))
        cost = engine.price(event)
        # Would have been $2.00 at the full input rate; paid $0.50.
        assert cost.uncached_equivalent == Decimal("2.0000000000")
        assert cost.cache_savings == Decimal("1.5000000000")

    def test_semantic_cache_hit_prices_the_avoided_call(self, engine: CostEngine) -> None:
        """The whole provider call was skipped, so the counterfactual is full price."""
        event = make_event(
            tokens=TokenUsage(input=1_000_000, output=1_000_000),
            served_from_semantic_cache=True,
        )
        cost = engine.price(event)
        assert cost.uncached_equivalent == Decimal("10.0000000000")

    def test_no_cache_means_no_phantom_savings(self, engine: CostEngine) -> None:
        cost = engine.price(make_event())
        assert cost.cache_savings == Decimal("0")


class TestSelfHosted:
    def test_gpu_time_is_priced_when_tokens_are_free(self) -> None:
        """Self-hosted cost is amortised GPU time, not metered tokens."""
        engine = CostEngine(infra_model=InfrastructureCostModel(gpu_hourly_rate=Decimal("3600")))
        event = make_event(
            provider=Provider.VLLM,
            model="llama-4-70b-local",
            tokens=TokenUsage(input=5000, output=1000),
            trace=RequestTrace(gpu_seconds=Decimal("10")),
        )
        cost = engine.price(event)
        assert cost.token_cost == Decimal("0")
        # 10 GPU-seconds at $3600/hour = $10 of compute.
        assert cost.compute_cost == Decimal("10.0000000000")
        assert cost.energy_cost > Decimal("0")
        assert cost.total > Decimal("10")

    def test_hosted_provider_ignores_gpu_seconds(self, engine: CostEngine) -> None:
        """A stray gpu_seconds value must not double-charge a hosted call."""
        event = make_event(trace=RequestTrace(gpu_seconds=Decimal("100")))
        assert engine.price(event).infrastructure_cost == Decimal("0")


class TestEffectiveDating:
    def test_historical_events_price_at_the_rate_in_force_then(self) -> None:
        """Re-running last quarter's chargeback must not restate history."""
        old = RateCard(
            provider=Provider.OPENAI,
            model="test-model",
            model_type=default_catalog.all_cards()[0].model_type,
            rates={TokenClass.INPUT: Decimal("10")},
            effective_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            effective_to=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        new = RateCard(
            provider=Provider.OPENAI,
            model="test-model",
            model_type=old.model_type,
            rates={TokenClass.INPUT: Decimal("2")},
            effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        engine = CostEngine(catalog=PricingCatalog([old, new]))

        historical = make_event(
            model="test-model",
            tokens=TokenUsage(input=1_000_000, output=0),
            occurred_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )
        current = make_event(
            model="test-model",
            tokens=TokenUsage(input=1_000_000, output=0),
            occurred_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        assert engine.price(historical).total == Decimal("10.0000000000")
        assert engine.price(current).total == Decimal("2.0000000000")

    def test_rate_card_version_is_recorded_for_audit(self, engine: CostEngine) -> None:
        cost = engine.price(make_event())
        assert cost.rate_card_version is not None
        assert "gpt-4.1" in cost.rate_card_version


class TestDiscounts:
    def test_enterprise_discount_applies_multiplicatively(self) -> None:
        card = RateCard(
            provider=Provider.OPENAI,
            model="discounted",
            model_type=default_catalog.all_cards()[0].model_type,
            rates={TokenClass.INPUT: Decimal("10")},
            discount_multiplier=Decimal("0.85"),
        )
        engine = CostEngine(catalog=PricingCatalog([card]))
        cost = engine.price(make_event(model="discounted", tokens=TokenUsage(input=1_000_000, output=0)))
        assert cost.total == Decimal("8.5000000000")


class TestPrecision:
    def test_no_float_drift_across_large_aggregations(self, engine: CostEngine) -> None:
        """Decimal is exact under summation; float accumulates drift at 10^5+."""
        events = [make_event(tokens=TokenUsage(input=333, output=111)) for _ in range(100_000)]
        costs = engine.price_many(events)
        total = sum(c.total for c in costs)
        single = costs[0].total
        assert total == single * 100_000

    def test_summarise_costs_reconciles_components(self, engine: CostEngine) -> None:
        events = [make_event(tokens=TokenUsage(input=1000, output=200, cached_input=500)) for _ in range(50)]
        costs = engine.price_many(events)
        summary = summarise_costs(costs)
        assert summary["total_cost"] == sum(c.total for c in costs)
        assert (
            summary["token_cost"] + summary["infrastructure_cost"] + summary["request_fees"]
            == summary["total_cost"]
        )


class TestEstimation:
    def test_estimate_matches_actual_pricing(self, engine: CostEngine) -> None:
        """The simulator's estimate must agree with what ingestion will charge."""
        tokens = TokenUsage(input=4000, output=800)
        estimated = engine.estimate(provider=Provider.OPENAI, model="gpt-4.1", tokens=tokens)
        actual = engine.price(make_event(tokens=tokens)).total
        assert estimated == actual
