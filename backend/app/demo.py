"""Synthetic workload generator.

Seeds a realistic tenant so every dashboard, detector and recommendation has
something to show without connecting real provider traffic.

The generator deliberately plants the specific pathologies the platform exists
to find, so a demo exercises the detectors rather than showing empty states:

- a chatbot whose context grows every turn (context explosion),
- an agent that fails to terminate (runaway agent),
- a service retrying a rate-limited provider (retry storm),
- a FAQ endpoint re-sending identical prompts (cache miss),
- a summarisation job on an expensive model doing trivial work (downgrade),
- a RAG pipeline with a 4k-token static preamble and no prompt cache.

Traffic follows a weekday/weekend and business-hours shape so the forecaster's
weekly seasonality is real rather than assumed.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from app.domain.enums import ModelType, Provider, RequestStatus
from app.domain.usage import AttributionContext, RequestTrace, TokenUsage, UsageEvent
from app.store import DEMO_ORG_ID, AnalyticsStore

#: Fixed ids so seeded data is stable across runs and the UI can deep-link.
DEPARTMENTS = {
    "Customer Support": UUID("11111111-0000-0000-0000-000000000001"),
    "Engineering": UUID("11111111-0000-0000-0000-000000000002"),
    "Sales": UUID("11111111-0000-0000-0000-000000000003"),
    "Legal": UUID("11111111-0000-0000-0000-000000000004"),
    "Marketing": UUID("11111111-0000-0000-0000-000000000005"),
}

TEAMS = {
    "support-platform": (UUID("22222222-0000-0000-0000-000000000001"), "Customer Support"),
    "developer-tools": (UUID("22222222-0000-0000-0000-000000000002"), "Engineering"),
    "search-rag": (UUID("22222222-0000-0000-0000-000000000003"), "Engineering"),
    "revenue-ai": (UUID("22222222-0000-0000-0000-000000000004"), "Sales"),
    "contract-review": (UUID("22222222-0000-0000-0000-000000000005"), "Legal"),
    "content-studio": (UUID("22222222-0000-0000-0000-000000000006"), "Marketing"),
}

COST_CENTERS = {
    "Customer Support": "CC-4100",
    "Engineering": "CC-2200",
    "Sales": "CC-3300",
    "Legal": "CC-5500",
    "Marketing": "CC-6600",
}


def _hourly_weight(moment: datetime) -> float:
    """Traffic shape: weekday business hours dominate.

    Without this the forecaster has no seasonality to find and the demo
    misrepresents how the model behaves on real data.
    """
    if moment.weekday() >= 5:
        return 0.18
    hour = moment.hour
    if 9 <= hour <= 17:
        return 1.0
    if 7 <= hour <= 20:
        return 0.45
    return 0.08


def _event(
    *,
    provider: Provider,
    model: str,
    team: str,
    feature: str,
    occurred_at: datetime,
    input_tokens: int,
    output_tokens: int,
    model_type: ModelType = ModelType.CHAT,
    cached_input: int = 0,
    reasoning: int = 0,
    status: RequestStatus = RequestStatus.SUCCESS,
    latency_ms: int = 900,
    retry_count: int = 0,
    parent_request_id: str | None = None,
    agent_run_id: str | None = None,
    agent_step: int = 0,
    conversation_id: str | None = None,
    conversation_turn: int = 0,
    system_prompt_tokens: int = 0,
    few_shot_tokens: int = 0,
    tool_definition_tokens: int = 0,
    rag_tokens: int = 0,
    rag_chunks: int = 0,
    fingerprint: str | None = None,
    application: str | None = None,
) -> UsageEvent:
    team_id, department = TEAMS[team]
    return UsageEvent(
        provider=provider,
        model=model,
        model_type=model_type,
        status=status,
        occurred_at=occurred_at,
        tokens=TokenUsage(
            input=input_tokens,
            output=output_tokens,
            cached_input=cached_input,
            reasoning=reasoning,
        ),
        attribution=AttributionContext(
            organization_id=DEMO_ORG_ID,
            department_id=DEPARTMENTS[department],
            team_id=team_id,
            user_id=uuid4(),
            feature=feature,
            application=application or team,
            environment="production",
            cost_center=COST_CENTERS[department],
        ),
        trace=RequestTrace(
            latency_ms=latency_ms,
            retry_count=retry_count,
            parent_request_id=parent_request_id,
            agent_run_id=agent_run_id,
            agent_step=agent_step,
            conversation_id=conversation_id,
            conversation_turn=conversation_turn,
            system_prompt_tokens=system_prompt_tokens,
            few_shot_tokens=few_shot_tokens,
            tool_definition_tokens=tool_definition_tokens,
            rag_tokens=rag_tokens,
            rag_chunks=rag_chunks,
        ),
        prompt_fingerprint=fingerprint,
    )


def seed(
    store: AnalyticsStore,
    *,
    days: int = 90,
    organization_id: UUID = DEMO_ORG_ID,
    seed_value: int = 20260804,
) -> int:
    """Populate `store` with `days` of synthetic traffic. Returns event count."""
    rng = random.Random(seed_value)
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days)
    count = 0

    # Growth factor so the series trends upward — adoption curves are the norm,
    # and a flat series would make the forecast trivially correct.
    for day_offset in range(days):
        day = start + timedelta(days=day_offset)
        growth = 1.0 + (day_offset / days) * 0.9

        for hour in range(24):
            moment = day + timedelta(hours=hour)
            weight = _hourly_weight(moment) * growth
            if weight < 0.05:
                continue

            # ---- Support chatbot: RAG + growing conversation context ----
            for _ in range(int(rng.gauss(14, 4) * weight)):
                turn = rng.randint(1, 12)
                # Context grows linearly with turn: history resent verbatim.
                history = 400 * turn
                rag = rng.randint(2500, 4200)
                count += _add(
                    store, organization_id,
                    _event(
                        provider=Provider.OPENAI, model="gpt-4.1",
                        team="support-platform", feature="support-chat",
                        occurred_at=moment + timedelta(minutes=rng.randint(0, 59)),
                        input_tokens=1800 + history + rag,
                        output_tokens=rng.randint(180, 420),
                        system_prompt_tokens=1800, rag_tokens=rag, rag_chunks=10,
                        conversation_id=f"conv-{day_offset}-{rng.randint(1, 60)}",
                        conversation_turn=turn,
                        latency_ms=rng.randint(700, 2100),
                        fingerprint=f"support-{rng.randint(1, 400)}",
                    ),
                )

            # ---- FAQ endpoint: heavy prompt repetition, no cache ----
            for _ in range(int(rng.gauss(9, 3) * weight)):
                count += _add(
                    store, organization_id,
                    _event(
                        provider=Provider.OPENAI, model="gpt-4.1",
                        team="support-platform", feature="faq-answering",
                        occurred_at=moment + timedelta(minutes=rng.randint(0, 59)),
                        input_tokens=2400, output_tokens=rng.randint(90, 200),
                        system_prompt_tokens=2000,
                        latency_ms=rng.randint(400, 900),
                        # Small fingerprint space => the same question over and
                        # over, which is exactly what a response cache captures.
                        fingerprint=f"faq-{rng.randint(1, 12)}",
                    ),
                )

            # ---- Code assistant: expensive reasoning model ----
            for _ in range(int(rng.gauss(6, 2) * weight)):
                count += _add(
                    store, organization_id,
                    _event(
                        provider=Provider.ANTHROPIC, model="claude-fable-5",
                        model_type=ModelType.REASONING,
                        team="developer-tools", feature="code-review",
                        occurred_at=moment + timedelta(minutes=rng.randint(0, 59)),
                        input_tokens=rng.randint(9000, 26000),
                        output_tokens=rng.randint(600, 1800),
                        reasoning=rng.randint(800, 3200),
                        system_prompt_tokens=3200, tool_definition_tokens=1400,
                        latency_ms=rng.randint(1800, 5200),
                    ),
                )

            # ---- Doc search: RAG with an oversized static preamble ----
            for _ in range(int(rng.gauss(11, 3) * weight)):
                rag = rng.randint(4000, 7000)
                count += _add(
                    store, organization_id,
                    _event(
                        provider=Provider.ANTHROPIC, model="claude-sonnet-5",
                        team="search-rag", feature="doc-search",
                        occurred_at=moment + timedelta(minutes=rng.randint(0, 59)),
                        input_tokens=4200 + rag,
                        output_tokens=rng.randint(220, 700),
                        system_prompt_tokens=3400, few_shot_tokens=800,
                        rag_tokens=rag, rag_chunks=12,
                        latency_ms=rng.randint(600, 1800),
                        fingerprint=f"search-{rng.randint(1, 900)}",
                    ),
                )

            # ---- Embeddings: continuous ingestion, some re-embedding ----
            for _ in range(int(rng.gauss(7, 2) * weight)):
                event = _event(
                    provider=Provider.OPENAI, model="text-embedding-3-large",
                    model_type=ModelType.EMBEDDING,
                    team="search-rag", feature="doc-indexing",
                    occurred_at=moment + timedelta(minutes=rng.randint(0, 59)),
                    input_tokens=0, output_tokens=0,
                    latency_ms=rng.randint(60, 200),
                    fingerprint=f"doc-{rng.randint(1, 260)}",
                )
                event.tokens.embedding = rng.randint(6000, 22000)
                count += _add(store, organization_id, event)

            # ---- Sales summarisation: trivial task on a premium model ----
            for _ in range(int(rng.gauss(5, 2) * weight)):
                count += _add(
                    store, organization_id,
                    _event(
                        provider=Provider.OPENAI, model="gpt-5",
                        model_type=ModelType.REASONING,
                        team="revenue-ai", feature="call-summary",
                        occurred_at=moment + timedelta(minutes=rng.randint(0, 59)),
                        input_tokens=rng.randint(3000, 7000),
                        output_tokens=rng.randint(150, 350),
                        reasoning=rng.randint(200, 900),
                        system_prompt_tokens=900,
                        latency_ms=rng.randint(1400, 3600),
                    ),
                )

            # ---- Contract review: low volume, very high value ----
            if rng.random() < 0.35 * weight:
                count += _add(
                    store, organization_id,
                    _event(
                        provider=Provider.ANTHROPIC, model="claude-opus-5",
                        model_type=ModelType.REASONING,
                        team="contract-review", feature="contract-analysis",
                        occurred_at=moment + timedelta(minutes=rng.randint(0, 59)),
                        input_tokens=rng.randint(40000, 120000),
                        output_tokens=rng.randint(1500, 4000),
                        reasoning=rng.randint(2000, 6000),
                        system_prompt_tokens=5200,
                        latency_ms=rng.randint(4000, 12000),
                    ),
                )

            # ---- Marketing copy: cheap model, high volume ----
            for _ in range(int(rng.gauss(8, 3) * weight)):
                count += _add(
                    store, organization_id,
                    _event(
                        provider=Provider.GOOGLE_GEMINI, model="gemini-2.5-flash",
                        team="content-studio", feature="copy-generation",
                        occurred_at=moment + timedelta(minutes=rng.randint(0, 59)),
                        input_tokens=rng.randint(800, 2200),
                        output_tokens=rng.randint(400, 1200),
                        system_prompt_tokens=600, few_shot_tokens=1100,
                        latency_ms=rng.randint(300, 900),
                    ),
                )

            # ---- Retry storm against a rate-limited provider ----
            if rng.random() < 0.5 * weight:
                parent = f"req-{day_offset}-{hour}"
                for attempt in range(rng.randint(2, 5)):
                    failed = attempt < 2
                    count += _add(
                        store, organization_id,
                        _event(
                            provider=Provider.GROQ, model="llama-4-70b",
                            team="developer-tools", feature="autocomplete",
                            occurred_at=moment + timedelta(minutes=attempt),
                            input_tokens=rng.randint(1200, 2600),
                            output_tokens=0 if failed else rng.randint(80, 260),
                            status=RequestStatus.RATE_LIMITED if failed else RequestStatus.SUCCESS,
                            retry_count=attempt,
                            parent_request_id=parent if attempt else None,
                            latency_ms=rng.randint(60, 400),
                        ),
                    )

    # ---- Runaway agent: a single recent run that never terminates ----
    run_id = f"agent-run-{uuid4().hex[:10]}"
    agent_start = now - timedelta(hours=6)
    for step in range(48):
        count += _add(
            store, organization_id,
            _event(
                provider=Provider.OPENAI, model="gpt-5",
                model_type=ModelType.REASONING,
                team="developer-tools", feature="autonomous-refactor",
                occurred_at=agent_start + timedelta(minutes=step * 4),
                # Context accumulates across steps — the agent is re-reading
                # its whole scratchpad every iteration.
                input_tokens=6000 + step * 900,
                output_tokens=rng.randint(400, 900),
                reasoning=rng.randint(600, 2000),
                agent_run_id=run_id, agent_step=step,
                system_prompt_tokens=2400, tool_definition_tokens=1800,
                latency_ms=rng.randint(2000, 6000),
            ),
        )

    # ---- Cost spike: a bad deploy that ran unbounded overnight ----
    spike_day = now - timedelta(days=2)
    for i in range(220):
        count += _add(
            store, organization_id,
            _event(
                provider=Provider.ANTHROPIC, model="claude-opus-5",
                model_type=ModelType.REASONING,
                team="content-studio", feature="bulk-rewrite",
                occurred_at=spike_day + timedelta(minutes=i * 2),
                input_tokens=rng.randint(28000, 60000),
                output_tokens=rng.randint(1200, 2600),
                reasoning=rng.randint(1000, 3000),
                system_prompt_tokens=1200,
                latency_ms=rng.randint(3000, 9000),
            ),
        )

    return count


def _add(store: AnalyticsStore, organization_id: UUID, event: UsageEvent) -> int:
    store.add_event(event, organization_id=organization_id)
    return 1


def seed_governance(store: AnalyticsStore, *, organization_id: UUID = DEMO_ORG_ID) -> None:
    """Add representative budgets, policies and entity display names.

    Budget amounts are derived from the spend actually seeded rather than
    hard-coded. A fixed $45k budget against a demo tenant spending $1k reads as
    0.1% utilised on every bar, which makes the whole budget-health surface —
    thresholds, severity colours, forecast-to-exceed warnings — invisible in a
    demo. Sizing them against observed spend puts the tenant in the band where
    those behaviours actually show.
    """
    from app.api.schemas import BudgetIn
    from app.domain.enums import BudgetScope, EnforcementAction
    from app.services.governance import Policy

    store.register_labels(
        {
            **{str(uid): name for name, uid in DEPARTMENTS.items()},
            **{str(uid): slug for slug, (uid, _department) in TEAMS.items()},
            str(organization_id): "Acme Corporation",
        },
        organization_id=organization_id,
    )

    def budget_for(target_utilisation: str, department: str | None = None) -> Decimal:
        spend = store.month_to_date_spend(
            organization_id,
            department_id=DEPARTMENTS[department] if department else None,
        )
        sized = (spend / Decimal(target_utilisation)).quantize(Decimal("1"))
        # Only fall back to a nominal amount when there is effectively no spend
        # to size against. A fixed floor applied unconditionally overrides the
        # target utilisation for every small department, which defeats the
        # point of deriving the budget from real spend in the first place.
        return sized if sized >= Decimal("1") else Decimal("50")

    # Three budgets landing in three different severity bands, so the
    # thresholds and their colours are all exercised on first load.
    store.create_budget(
        BudgetIn(
            name="Engineering monthly",
            scope="department",
            scope_id=str(DEPARTMENTS["Engineering"]),
            amount=budget_for("0.96", "Engineering"),
            period="monthly",
            action_at_limit="warn",
        ),
        organization_id=organization_id,
    )
    store.create_budget(
        BudgetIn(
            name="Customer Support monthly",
            scope="department",
            scope_id=str(DEPARTMENTS["Customer Support"]),
            amount=budget_for("0.55", "Customer Support"),
            period="monthly",
            action_at_limit="warn",
        ),
        organization_id=organization_id,
    )
    store.create_budget(
        BudgetIn(
            name="Organization monthly",
            scope="organization",
            scope_id=str(organization_id),
            amount=budget_for("0.82"),
            period="monthly",
            action_at_limit="require_approval",
            hard_stop_multiplier=Decimal("1.25"),
        ),
        organization_id=organization_id,
    )
    store.add_policy(
        Policy(
            name="Block oversized single requests",
            action=EnforcementAction.BLOCK,
            scope=BudgetScope.ORGANIZATION,
            max_cost_per_request=Decimal("25"),
            max_context_tokens=400_000,
            priority=200,
        ),
        organization_id=organization_id,
    )
    store.add_policy(
        Policy(
            name="Approval required for high-cost reasoning calls",
            action=EnforcementAction.REQUIRE_APPROVAL,
            scope=BudgetScope.ORGANIZATION,
            # Set below the $25 hard ceiling and reachable within the 400k
            # context cap, so both policies are actually exercisable against
            # real models rather than one shadowing the other.
            require_approval_above=Decimal("3"),
            priority=100,
        ),
        organization_id=organization_id,
    )


def main() -> None:  # pragma: no cover - CLI entrypoint
    from app.store import default_store

    total = seed(default_store)
    seed_governance(default_store)
    print(f"seeded {total:,} usage events for organization {DEMO_ORG_ID}")


if __name__ == "__main__":  # pragma: no cover
    main()
