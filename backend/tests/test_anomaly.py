"""Anomaly and waste detection.

Covers every `AnomalyKind` the platform declares. Before this file existed,
seven of the twelve kinds (`TOKEN_SPIKE`, `LATENCY_SPIKE`, `INFINITE_LOOP`,
`PROVIDER_DRIFT`, `PROMPT_INJECTION`, `API_ABUSE`, `RAG_MISCONFIGURATION`)
were declared in the enum but never emitted by any detector — this both
implements and locks in the missing five detectors plus the two structural
ones (`RUNAWAY_AGENT`, `RETRY_STORM`, `CONTEXT_EXPLOSION`, `ERROR_BURST`,
`COST_SPIKE`) that had no dedicated tests at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID, uuid4

from app.domain.enums import AnomalyKind, Provider, RequestStatus
from app.domain.usage import AttributionContext, CostBreakdown, RequestTrace, TokenUsage, UsageEvent
from app.services import anomaly as svc

FLAT_USER = UUID("00000000-0000-0000-0000-000000000001")


def make_event(**kwargs) -> UsageEvent:
    defaults: dict = {
        "provider": Provider.OPENAI,
        "model": "gpt-4.1",
        "tokens": TokenUsage(input=1000, output=500),
    }
    defaults.update(kwargs)
    return UsageEvent(**defaults)


def make_cost(event: UsageEvent, total: Decimal) -> CostBreakdown:
    return CostBreakdown(event_id=event.id, request_fee=total)


def costs_for(pairs: list[tuple[UsageEvent, Decimal]]) -> dict[str, CostBreakdown]:
    return {str(e.id): make_cost(e, c) for e, c in pairs}


def flat_then_spike(
    days: int = 6, base: Decimal = Decimal("100"), spike: Decimal = Decimal("2000")
) -> list[Decimal]:
    return [base] * days + [spike]


class TestRobustZscore:
    def test_flat_series_falls_back_to_ratio(self) -> None:
        score, expected = svc.robust_zscore(Decimal("900"), [Decimal("100")] * 5)
        assert expected == Decimal("100")
        assert score > svc.DEFAULT_Z_THRESHOLD

    def test_short_history_returns_zero(self) -> None:
        score, _expected = svc.robust_zscore(Decimal("100"), [Decimal("100")])
        assert score == Decimal("0")


class TestDetectSeriesAnomaly:
    def test_flags_upward_spike(self) -> None:
        finding = svc.detect_series_anomaly(
            series=flat_then_spike(),
            kind=AnomalyKind.COST_SPIKE,
            scope="daily_cost",
            scope_key="openai/gpt-4.1",
        )
        assert finding is not None
        assert finding.kind is AnomalyKind.COST_SPIKE
        assert finding.estimated_impact > Decimal("0")

    def test_ignores_downward_movement(self) -> None:
        series = [Decimal("100")] * 6 + [Decimal("10")]
        finding = svc.detect_series_anomaly(
            series=series, kind=AnomalyKind.COST_SPIKE, scope="daily_cost", scope_key="k"
        )
        assert finding is None

    def test_too_short_returns_none(self) -> None:
        finding = svc.detect_series_anomaly(
            series=[Decimal("1"), Decimal("2")],
            kind=AnomalyKind.COST_SPIKE,
            scope="daily_cost",
            scope_key="k",
        )
        assert finding is None

    def test_impact_of_overrides_raw_delta(self) -> None:
        finding = svc.detect_series_anomaly(
            series=flat_then_spike(),
            kind=AnomalyKind.TOKEN_SPIKE,
            scope="model",
            scope_key="k",
            label="tokens",
            unit=" tokens",
            impact_of=lambda current, expected: (current - expected) * Decimal("0.002"),
        )
        assert finding is not None
        # (2000 - 100) * 0.002 == 3.80
        assert finding.estimated_impact == Decimal("3.80")


class TestRunawayAgents:
    def test_flags_deep_run(self) -> None:
        run_id = "run-1"
        events = [
            make_event(trace=RequestTrace(agent_run_id=run_id, agent_step=step)) for step in range(1, 31)
        ]
        costs = costs_for([(e, Decimal("2")) for e in events])
        findings = svc.detect_runaway_agents(events, costs)
        assert len(findings) == 1
        assert findings[0].kind is AnomalyKind.RUNAWAY_AGENT
        assert findings[0].scope_key == run_id

    def test_shallow_cheap_run_not_flagged(self) -> None:
        events = [
            make_event(trace=RequestTrace(agent_run_id="run-2", agent_step=step)) for step in range(1, 4)
        ]
        costs = costs_for([(e, Decimal("0.10")) for e in events])
        assert svc.detect_runaway_agents(events, costs) == []


class TestRetryStorms:
    def test_flags_high_retry_ratio(self) -> None:
        retried = [make_event(trace=RequestTrace(retry_count=1, parent_request_id="p")) for _ in range(20)]
        clean = [make_event() for _ in range(60)]
        events = retried + clean
        costs = costs_for([(e, Decimal("1")) for e in retried])
        findings = svc.detect_retry_storms(events, costs)
        assert len(findings) == 1
        assert findings[0].kind is AnomalyKind.RETRY_STORM

    def test_below_min_requests_not_flagged(self) -> None:
        events = [make_event(trace=RequestTrace(retry_count=1)) for _ in range(5)]
        assert svc.detect_retry_storms(events, {}) == []


class TestContextExplosion:
    def test_flags_growing_conversation(self) -> None:
        cid = "conv-1"
        events = [
            make_event(
                tokens=TokenUsage(input=100 * (turn + 1), output=50),
                trace=RequestTrace(conversation_id=cid, conversation_turn=turn),
            )
            for turn in range(6)
        ]
        costs = costs_for([(e, Decimal("1")) for e in events])
        findings = svc.detect_context_explosion(events, costs)
        assert len(findings) == 1
        assert findings[0].kind is AnomalyKind.CONTEXT_EXPLOSION

    def test_flat_conversation_not_flagged(self) -> None:
        cid = "conv-2"
        events = [
            make_event(
                tokens=TokenUsage(input=100, output=50),
                trace=RequestTrace(conversation_id=cid, conversation_turn=turn),
            )
            for turn in range(6)
        ]
        assert svc.detect_context_explosion(events, {}) == []


class TestDuplicateRequests:
    def test_flags_repeated_fingerprint(self) -> None:
        events = [
            make_event(prompt_fingerprint="fp-1", attribution=AttributionContext(feature="faq"))
            for _ in range(10)
        ]
        costs = costs_for([(e, Decimal("0.50")) for e in events])
        findings = svc.detect_duplicate_requests(events, costs)
        assert len(findings) == 1
        assert findings[0].kind is AnomalyKind.COST_SPIKE
        assert findings[0].estimated_impact > Decimal("0")

    def test_semantic_cache_hits_excluded(self) -> None:
        events = [
            make_event(
                prompt_fingerprint="fp-2",
                served_from_semantic_cache=True,
                attribution=AttributionContext(feature="faq"),
            )
            for _ in range(10)
        ]
        assert svc.detect_duplicate_requests(events, {}) == []


class TestErrorBursts:
    def test_flags_elevated_error_rate(self) -> None:
        errors = [make_event(status=RequestStatus.ERROR) for _ in range(10)]
        ok = [make_event() for _ in range(30)]
        events = errors + ok
        costs = costs_for([(e, Decimal("0.20")) for e in errors])
        findings = svc.detect_error_bursts(events, costs)
        assert len(findings) == 1
        assert findings[0].kind is AnomalyKind.ERROR_BURST

    def test_healthy_provider_not_flagged(self) -> None:
        events = [make_event() for _ in range(40)]
        assert svc.detect_error_bursts(events, {}) == []


class TestTokenSpikes:
    def test_flags_token_spike_with_dollar_impact(self) -> None:
        tokens = {"openai/gpt-4.1": flat_then_spike(base=Decimal("10000"), spike=Decimal("500000"))}
        costs = {"openai/gpt-4.1": [Decimal("1")] * 6 + [Decimal("50")]}
        findings = svc.detect_token_spikes(tokens, costs)
        assert len(findings) == 1
        assert findings[0].kind is AnomalyKind.TOKEN_SPIKE
        assert findings[0].estimated_impact > Decimal("0")

    def test_flat_series_not_flagged(self) -> None:
        tokens = {"k": [Decimal("1000")] * 7}
        assert svc.detect_token_spikes(tokens, {}) == []


class TestLatencySpikes:
    def test_flags_latency_spike_with_zero_dollar_impact(self) -> None:
        latency = {"openai/gpt-4.1": flat_then_spike(base=Decimal("200"), spike=Decimal("9000"))}
        findings = svc.detect_latency_spikes(latency)
        assert len(findings) == 1
        assert findings[0].kind is AnomalyKind.LATENCY_SPIKE
        assert findings[0].estimated_impact == Decimal("0.00")

    def test_stable_latency_not_flagged(self) -> None:
        latency = {"k": [Decimal("200")] * 7}
        assert svc.detect_latency_spikes(latency) == []


class TestProviderDrift:
    def test_flags_unit_cost_jump(self) -> None:
        # $1 / 10k tokens for six days ($0.10/1k), then $1 / 1k tokens on the
        # seventh (10x the unit price) with volume above the noise floor.
        tokens = {"k": [Decimal("10000")] * 6 + [Decimal("10000")]}
        costs = {"k": [Decimal("1")] * 6 + [Decimal("10")]}
        findings = svc.detect_provider_drift(costs, tokens)
        assert len(findings) == 1
        assert findings[0].kind is AnomalyKind.PROVIDER_DRIFT
        assert findings[0].estimated_impact > Decimal("0")

    def test_stable_unit_cost_not_flagged(self) -> None:
        tokens = {"k": [Decimal("10000")] * 7}
        costs = {"k": [Decimal("1")] * 7}
        assert svc.detect_provider_drift(costs, tokens) == []

    def test_low_volume_day_skipped(self) -> None:
        # Today's volume is below the noise floor, so drift is not evaluated
        # even though the ratio moved.
        tokens = {"k": [Decimal("10000")] * 6 + [Decimal("10")]}
        costs = {"k": [Decimal("1")] * 6 + [Decimal("1")]}
        assert svc.detect_provider_drift(costs, tokens) == []


class TestInfiniteLoops:
    def test_flags_repeated_fingerprint_within_run(self) -> None:
        run_id = "run-stuck"
        events = [
            make_event(trace=RequestTrace(agent_run_id=run_id, agent_step=i), prompt_fingerprint="fp-x")
            for i in range(6)
        ]
        costs = costs_for([(e, Decimal("0.50")) for e in events])
        findings = svc.detect_infinite_loops(events, costs)
        assert len(findings) == 1
        assert findings[0].kind is AnomalyKind.INFINITE_LOOP
        assert findings[0].evidence["repeated_calls"] == 6

    def test_progressing_run_not_flagged(self) -> None:
        run_id = "run-fine"
        events = [
            make_event(trace=RequestTrace(agent_run_id=run_id, agent_step=i), prompt_fingerprint=f"fp-{i}")
            for i in range(6)
        ]
        assert svc.detect_infinite_loops(events, {}) == []


class TestPromptInjectionBursts:
    def test_flags_filtered_burst(self) -> None:
        uid = uuid4()
        filtered = [
            make_event(status=RequestStatus.FILTERED, attribution=AttributionContext(user_id=uid))
            for _ in range(12)
        ]
        ok = [make_event(attribution=AttributionContext(user_id=uid)) for _ in range(10)]
        events = filtered + ok
        costs = costs_for([(e, Decimal("0.05")) for e in filtered])
        findings = svc.detect_prompt_injection_bursts(events, costs)
        assert len(findings) == 1
        assert findings[0].kind is AnomalyKind.PROMPT_INJECTION

    def test_occasional_filter_not_flagged(self) -> None:
        uid = uuid4()
        events = [make_event(attribution=AttributionContext(user_id=uid)) for _ in range(25)]
        events[0] = make_event(status=RequestStatus.FILTERED, attribution=AttributionContext(user_id=uid))
        assert svc.detect_prompt_injection_bursts(events, {}) == []


class TestApiAbuse:
    def test_flags_burst_within_window(self) -> None:
        uid = uuid4()
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            make_event(
                attribution=AttributionContext(user_id=uid),
                occurred_at=base + timedelta(seconds=i * 0.3),
            )
            for i in range(150)
        ]
        costs = costs_for([(e, Decimal("0.01")) for e in events])
        findings = svc.detect_api_abuse(events, costs)
        assert len(findings) == 1
        assert findings[0].kind is AnomalyKind.API_ABUSE
        assert findings[0].evidence["peak_requests"] >= 120

    def test_spread_out_requests_not_flagged(self) -> None:
        uid = uuid4()
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            make_event(attribution=AttributionContext(user_id=uid), occurred_at=base + timedelta(hours=i))
            for i in range(150)
        ]
        assert svc.detect_api_abuse(events, {}) == []


class TestRagMisconfiguration:
    def test_flags_rag_heavy_prompts(self) -> None:
        events = [
            make_event(
                tokens=TokenUsage(input=1000, output=100),
                trace=RequestTrace(rag_chunks=20, rag_tokens=900),
                attribution=AttributionContext(feature="support-bot"),
            )
            for _ in range(25)
        ]
        costs = costs_for([(e, Decimal("1")) for e in events])
        findings = svc.detect_rag_misconfiguration(events, costs)
        assert len(findings) == 1
        assert findings[0].kind is AnomalyKind.RAG_MISCONFIGURATION

    def test_modest_rag_share_not_flagged(self) -> None:
        events = [
            make_event(
                tokens=TokenUsage(input=1000, output=100),
                trace=RequestTrace(rag_chunks=3, rag_tokens=100),
                attribution=AttributionContext(feature="support-bot"),
            )
            for _ in range(25)
        ]
        costs = costs_for([(e, Decimal("1")) for e in events])
        assert svc.detect_rag_misconfiguration(events, costs) == []


class TestDetectAll:
    def test_ranks_by_dollar_impact_and_wires_all_series(self) -> None:
        run_id = "run-3"
        agent_events = [
            make_event(trace=RequestTrace(agent_run_id=run_id, agent_step=step)) for step in range(1, 31)
        ]
        events = agent_events
        costs = costs_for([(e, Decimal("5")) for e in agent_events])

        # Cost jumps 100x while tokens jump only 10x, so unit price ($/1k
        # tokens) also jumps 10x — exercises PROVIDER_DRIFT alongside a plain
        # COST_SPIKE and TOKEN_SPIKE, which a proportional spike would not.
        cost_series = {"openai/gpt-4.1": flat_then_spike(base=Decimal("50"), spike=Decimal("5000"))}
        token_series = {"openai/gpt-4.1": flat_then_spike(base=Decimal("5000"), spike=Decimal("50000"))}
        latency_series = {"openai/gpt-4.1": flat_then_spike(base=Decimal("200"), spike=Decimal("8000"))}

        findings = svc.detect_all(
            events,
            costs,
            daily_cost_series=cost_series,
            daily_token_series=token_series,
            daily_latency_series=latency_series,
        )

        kinds = {f.kind for f in findings}
        assert AnomalyKind.RUNAWAY_AGENT in kinds
        assert AnomalyKind.COST_SPIKE in kinds
        assert AnomalyKind.TOKEN_SPIKE in kinds
        assert AnomalyKind.LATENCY_SPIKE in kinds
        assert AnomalyKind.PROVIDER_DRIFT in kinds

        impacts = [f.estimated_impact for f in findings]
        assert impacts == sorted(impacts, reverse=True)

    def test_accepts_list_of_cost_breakdowns(self) -> None:
        events = [make_event() for _ in range(3)]
        costs = [CostBreakdown(event_id=e.id, request_fee=Decimal("1")) for e in events]
        # Should not raise when costs is a list rather than a dict.
        svc.detect_all(events, costs)

    def test_no_optional_series_still_runs_structural_detectors(self) -> None:
        events = [make_event() for _ in range(3)]
        findings = svc.detect_all(events, {})
        assert isinstance(findings, list)
