"""In-memory analytics store.

## Why this exists

The production data path is Postgres (partitioned facts + rollups) fed by Kafka
and Celery. That stack is defined in `app/db/models.py`, `docker-compose.yml`
and the Kubernetes manifests. But requiring all of it to be running before
anyone can see the product is a real adoption cost — for local development,
for CI, for demos, and for the API contract tests.

`AnalyticsStore` implements the same query surface the routers depend on,
backed by plain Python collections. `make demo` seeds it with synthetic usage
and the whole platform is explorable with no external services. The SQL-backed
implementation satisfies the same interface, so swapping it is a change to one
dependency provider in `deps.py` and nothing else.

This is a deliberate seam, not a stub: every method here defines the contract
the repository layer must honour, including the aggregation semantics that are
easy to get subtly wrong in SQL (dense date series, unattributed bucketing,
shared-cost allocation).
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from app.api import schemas as s
from app.domain.enums import (
    BudgetPeriod,
    BudgetScope,
    EnforcementAction,
    ModelType,
    Provider,
    RequestStatus,
)
from app.domain.money import ZERO, quantize_cost, safe_div
from app.domain.usage import (
    AttributionContext,
    CostBreakdown,
    RequestTrace,
    UsageEvent,
    coerce_tokens,
)
from app.services.cost_engine import CostEngine, UnknownModelError
from app.services.governance import (
    Budget,
    BudgetStatus,
    Policy,
    PolicyEngine,
    RequestIntent,
    build_chargeback,
)


class AnalyticsStore:
    """Thread-safe in-memory implementation of the analytics query surface."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: dict[UUID, list[UsageEvent]] = defaultdict(list)
        self._costs: dict[str, CostBreakdown] = {}
        self._idempotency: dict[UUID, set[str]] = defaultdict(set)
        self._budgets: dict[UUID, list[Budget]] = defaultdict(list)
        self._policies: dict[UUID, list[Policy]] = defaultdict(list)
        #: Entity id -> human label, for departments and teams. Facts are keyed
        #: on immutable ids (a team can be renamed without restating history),
        #: but a dashboard showing raw UUIDs is unreadable, so the display name
        #: is resolved at the API boundary. In production this is a cached
        #: lookup over the departments/teams tables.
        self._labels: dict[UUID, dict[str, str]] = defaultdict(dict)
        self._engine = CostEngine()

    # -- ingestion ----------------------------------------------------------

    def ingest(
        self, payload: list[s.UsageEventIn], *, organization_id: UUID
    ) -> s.IngestResult:
        result = s.IngestResult(accepted=0)
        with self._lock:
            seen = self._idempotency[organization_id]
            for item in payload:
                if item.idempotency_key and item.idempotency_key in seen:
                    result.duplicates += 1
                    continue
                try:
                    event = self._to_domain(item, organization_id=organization_id)
                except ValueError as exc:
                    result.rejected += 1
                    result.errors.append(str(exc))
                    continue

                try:
                    cost = self._engine.price(event, strict=True)
                except UnknownModelError as exc:
                    # Accepted but flagged. Silently zero-pricing an unknown
                    # model is how a platform under-reports the newest, most
                    # expensive model a team just adopted.
                    cost = self._engine.price(event, strict=False)
                    result.unpriced += 1
                    result.errors.append(str(exc))

                self._events[organization_id].append(event)
                self._costs[str(event.id)] = cost
                if item.idempotency_key:
                    seen.add(item.idempotency_key)
                result.accepted += 1
        return result

    def add_event(self, event: UsageEvent, *, organization_id: UUID) -> CostBreakdown:
        """Direct insertion, used by the demo seeder and tests."""
        cost = self._engine.price(event, strict=False)
        with self._lock:
            self._events[organization_id].append(event)
            self._costs[str(event.id)] = cost
        return cost

    def _to_domain(self, item: s.UsageEventIn, *, organization_id: UUID) -> UsageEvent:
        try:
            provider = Provider(item.provider)
        except ValueError as exc:
            raise ValueError(f"unknown provider '{item.provider}'") from exc
        try:
            model_type = ModelType(item.model_type)
        except ValueError:
            model_type = ModelType.CHAT
        try:
            request_status = RequestStatus(item.status)
        except ValueError:
            request_status = RequestStatus.SUCCESS

        trace = item.trace
        return UsageEvent(
            provider=provider,
            model=item.model,
            tokens=coerce_tokens(item.tokens.model_dump()),
            idempotency_key=item.idempotency_key,
            model_type=model_type,
            status=request_status,
            occurred_at=item.occurred_at or datetime.now(UTC),
            attribution=AttributionContext(
                organization_id=organization_id,
                department_id=item.attribution.department_id,
                team_id=item.attribution.team_id,
                user_id=item.attribution.user_id,
                project=item.attribution.project,
                feature=item.attribution.feature,
                application=item.attribution.application,
                environment=item.attribution.environment,
                customer_id=item.attribution.customer_id,
                cost_center=item.attribution.cost_center,
                tags=item.attribution.tags,
            ),
            trace=RequestTrace(
                latency_ms=trace.latency_ms,
                time_to_first_token_ms=trace.time_to_first_token_ms,
                streamed=trace.streamed,
                retry_count=trace.retry_count,
                parent_request_id=trace.parent_request_id,
                agent_run_id=trace.agent_run_id,
                agent_step=trace.agent_step,
                conversation_id=trace.conversation_id,
                conversation_turn=trace.conversation_turn,
                rag_chunks=trace.rag_chunks,
                rag_tokens=trace.rag_tokens,
                system_prompt_tokens=trace.system_prompt_tokens,
                few_shot_tokens=trace.few_shot_tokens,
                tool_definition_tokens=trace.tool_definition_tokens,
                gpu_seconds=trace.gpu_seconds,
                network_gb=trace.network_gb,
            ),
            prompt_template_id=item.prompt_template_id,
            prompt_version=item.prompt_version,
            prompt_fingerprint=item.prompt_fingerprint,
            served_from_semantic_cache=item.served_from_semantic_cache,
            error_code=item.error_code,
            metadata=item.metadata,
        )

    # -- queries ------------------------------------------------------------

    def events_between(
        self, organization_id: UUID, start: datetime, end: datetime
    ) -> list[UsageEvent]:
        with self._lock:
            return [
                e for e in self._events.get(organization_id, [])
                if start <= e.occurred_at <= end
            ]

    def register_labels(self, labels: dict[str, str], *, organization_id: UUID) -> None:
        """Register display names for entity ids (departments, teams, …)."""
        with self._lock:
            self._labels[organization_id].update(labels)

    def label_for(self, organization_id: UUID, key: str) -> str:
        """Resolve an entity id to its display name, falling back to the key.

        Falling back rather than raising: an id with no registered label is a
        normal condition (a team created after the last catalog refresh), and a
        dashboard should degrade to showing the id, not fail to render.
        """
        return self._labels.get(organization_id, {}).get(key, key)

    def cost_of(self, event: UsageEvent) -> CostBreakdown:
        return self._costs.get(str(event.id), CostBreakdown(event_id=event.id))

    def costs_for(self, events: list[UsageEvent]) -> dict[str, CostBreakdown]:
        return {str(e.id): self.cost_of(e) for e in events}

    def daily_series(
        self, organization_id: UUID, start: datetime, end: datetime
    ) -> list[s.TimeSeriesPoint]:
        events = self.events_between(organization_id, start, end)
        buckets: dict[date, dict[str, Decimal | int]] = defaultdict(
            lambda: {"cost": ZERO, "tokens": 0, "requests": 0}
        )
        for event in events:
            bucket = buckets[event.occurred_at.date()]
            bucket["cost"] = bucket["cost"] + self.cost_of(event).total  # type: ignore[operator]
            bucket["tokens"] = int(bucket["tokens"]) + event.tokens.total
            bucket["requests"] = int(bucket["requests"]) + 1

        # Dense series: emit every day in range, zero-filled. A sparse series
        # makes a chart misrepresent a gap as a straight line between two
        # distant points, and breaks the forecaster's spacing assumption.
        out: list[s.TimeSeriesPoint] = []
        cursor = start.date()
        while cursor <= end.date():
            bucket = buckets.get(cursor, {"cost": ZERO, "tokens": 0, "requests": 0})
            out.append(
                s.TimeSeriesPoint(
                    at=cursor,
                    cost=quantize_cost(bucket["cost"]),  # type: ignore[arg-type]
                    tokens=int(bucket["tokens"]),
                    requests=int(bucket["requests"]),
                )
            )
            cursor += timedelta(days=1)
        return out

    def dense_daily_costs(
        self, organization_id: UUID, start: datetime, end: datetime
    ) -> list[Decimal]:
        return [p.cost for p in self.daily_series(organization_id, start, end)]

    def daily_cost_series_by_model(
        self, organization_id: UUID, start: datetime, end: datetime
    ) -> dict[str, list[Decimal]]:
        """Per-model dense daily cost, for the series anomaly detectors."""
        events = self.events_between(organization_id, start, end)
        by_model: dict[str, dict[date, Decimal]] = defaultdict(lambda: defaultdict(lambda: ZERO))
        for event in events:
            key = f"{event.provider}/{event.model}"
            by_model[key][event.occurred_at.date()] += self.cost_of(event).total

        days: list[date] = []
        cursor = start.date()
        while cursor <= end.date():
            days.append(cursor)
            cursor += timedelta(days=1)
        return {model: [buckets.get(d, ZERO) for d in days] for model, buckets in by_model.items()}

    def month_to_date_spend(
        self, organization_id: UUID, *, department_id: UUID | None = None
    ) -> Decimal:
        """Spend so far in the current calendar month, optionally per department.

        Matches the window a monthly budget is evaluated against, so a budget
        sized from this figure produces the utilisation the caller intended.
        """
        now = datetime.now(UTC)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        events = self.events_between(organization_id, start, now)
        if department_id is not None:
            events = [e for e in events if e.attribution.department_id == department_id]
        return sum((self.cost_of(e).total for e in events), ZERO)

    # -- governance ---------------------------------------------------------

    def create_budget(self, payload: s.BudgetIn, *, organization_id: UUID) -> s.BudgetStatusOut:
        budget = Budget(
            name=payload.name,
            scope=BudgetScope(payload.scope),
            scope_id=payload.scope_id,
            amount=payload.amount,
            period=BudgetPeriod(payload.period),
            alert_thresholds=tuple(payload.alert_thresholds),
            action_at_limit=EnforcementAction(payload.action_at_limit),
            hard_stop_multiplier=payload.hard_stop_multiplier,
            rollover=payload.rollover,
        )
        with self._lock:
            self._budgets[organization_id].append(budget)
        return self._budget_status_out(budget, organization_id)

    def budget_statuses(self, organization_id: UUID) -> list[s.BudgetStatusOut]:
        with self._lock:
            budgets = list(self._budgets.get(organization_id, []))
        return [self._budget_status_out(b, organization_id) for b in budgets]

    def _spent_in_period(self, organization_id: UUID, budget: Budget) -> Decimal:
        start, end = budget.period_bounds()
        events = self.events_between(
            organization_id,
            datetime.combine(start, datetime.min.time(), tzinfo=UTC),
            datetime.combine(end, datetime.max.time(), tzinfo=UTC),
        )
        matching = [e for e in events if self._matches_scope(e, budget)]
        return sum((self.cost_of(e).total for e in matching), ZERO)

    def _matches_scope(self, event: UsageEvent, budget: Budget) -> bool:
        if budget.scope is BudgetScope.ORGANIZATION:
            return True
        attribution = event.attribution
        mapping: dict[BudgetScope, str | None] = {
            BudgetScope.DEPARTMENT: str(attribution.department_id) if attribution.department_id else None,
            BudgetScope.TEAM: str(attribution.team_id) if attribution.team_id else None,
            BudgetScope.USER: str(attribution.user_id) if attribution.user_id else None,
            BudgetScope.PROJECT: attribution.project,
            BudgetScope.FEATURE: attribution.feature,
            BudgetScope.MODEL: event.model,
            BudgetScope.PROVIDER: str(event.provider),
        }
        return mapping.get(budget.scope) == budget.scope_id

    def _budget_status_out(self, budget: Budget, organization_id: UUID) -> s.BudgetStatusOut:
        start, end = budget.period_bounds()
        spent = self._spent_in_period(organization_id, budget)
        status = BudgetStatus(
            budget=budget, spent=spent, period_start=start, period_end=end
        )
        return s.BudgetStatusOut(
            id=budget.id,
            name=budget.name or budget.scope_id,
            scope=str(budget.scope),
            scope_id=budget.scope_id,
            amount=budget.amount,
            spent=quantize_cost(spent),
            remaining=quantize_cost(status.remaining),
            utilisation_pct=status.utilisation * Decimal("100"),
            period=str(budget.period),
            period_start=start,
            period_end=end,
            severity=str(status.severity),
            is_exceeded=status.is_exceeded,
            projected_to_exceed=status.projected_to_exceed,
        )

    def evaluate_preflight(
        self, payload: s.PreflightIn, *, organization_id: UUID
    ) -> s.PreflightOut:
        try:
            provider = Provider(payload.provider)
        except ValueError:
            # An unknown provider is not a reason to block a customer's
            # production request. Allow, and let ingestion flag it.
            return s.PreflightOut(
                action=str(EnforcementAction.ALLOW),
                allowed=True,
                estimated_cost=ZERO,
                reasons=[f"unknown provider '{payload.provider}'; not evaluated"],
                evaluated_in_ms=0.0,
            )

        from app.domain.usage import TokenUsage

        try:
            estimated = self._engine.estimate(
                provider=provider,
                model=payload.model,
                tokens=TokenUsage(
                    input=payload.estimated_input_tokens,
                    output=payload.estimated_output_tokens,
                ),
            )
        except UnknownModelError:
            estimated = ZERO

        intent = RequestIntent(
            provider=payload.provider,
            model=payload.model,
            estimated_tokens=payload.estimated_input_tokens + payload.estimated_output_tokens,
            estimated_cost=estimated,
            context_tokens=payload.estimated_input_tokens,
            department_id=payload.department_id,
            team_id=payload.team_id,
            user_id=payload.user_id,
            feature=payload.feature,
            environment=payload.environment,
        )

        with self._lock:
            policies = list(self._policies.get(organization_id, []))
            budgets = list(self._budgets.get(organization_id, []))

        budget_status = None
        for budget in budgets:
            start, end = budget.period_bounds()
            spent = self._spent_in_period(organization_id, budget)
            candidate = BudgetStatus(
                budget=budget, spent=spent, period_start=start, period_end=end
            )
            # Evaluate against the most-utilised applicable budget: the
            # tightest constraint is the one that governs.
            if budget_status is None or candidate.utilisation > budget_status.utilisation:
                budget_status = candidate

        decision = PolicyEngine(policies).evaluate_request(intent, budget_status=budget_status)
        return s.PreflightOut(
            action=str(decision.action),
            allowed=decision.allowed,
            estimated_cost=quantize_cost(estimated),
            reasons=decision.reasons,
            violated_policies=decision.violated_policies,
            suggested_model=decision.suggested_model,
            evaluated_in_ms=decision.evaluated_in_ms,
        )

    def add_policy(self, policy: Policy, *, organization_id: UUID) -> None:
        with self._lock:
            self._policies[organization_id].append(policy)

    def chargeback(
        self, organization_id: UUID, start: datetime, end: datetime
    ) -> list[s.ChargebackLineOut]:
        events = self.events_between(organization_id, start, end)
        direct: dict[str, Decimal] = defaultdict(lambda: ZERO)
        tokens: dict[str, int] = defaultdict(int)
        requests: dict[str, int] = defaultdict(int)
        shared = ZERO

        for event in events:
            cost = self.cost_of(event).total
            centre = event.attribution.cost_center
            if not centre:
                # Unattributed spend becomes shared cost, allocated
                # proportionally. Assigning it to a catch-all "unknown" cost
                # centre instead would let teams avoid chargeback simply by not
                # instrumenting, which is precisely the wrong incentive.
                shared += cost
                continue
            direct[centre] += cost
            tokens[centre] += event.tokens.total
            requests[centre] += 1

        lines = build_chargeback(
            dict(direct),
            shared_cost=shared,
            token_counts=dict(tokens),
            request_counts=dict(requests),
        )
        total = sum((line.total for line in lines), ZERO)
        return [
            s.ChargebackLineOut(
                cost_center=line.cost_center,
                department=line.department,
                direct_cost=line.direct_cost,
                allocated_shared_cost=line.allocated_shared_cost,
                total=line.total,
                tokens=line.tokens,
                requests=line.requests,
                share_pct=safe_div(line.total, total) * Decimal("100"),
            )
            for line in lines
        ]

    # -- admin --------------------------------------------------------------

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._costs.clear()
            self._idempotency.clear()
            self._budgets.clear()
            self._policies.clear()

    @property
    def event_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._events.values())


#: Process-wide default store. Replaced by the SQL repository in production
#: via the dependency override in `deps.py`.
default_store = AnalyticsStore()

#: Fixed demo tenant so seeded data and API calls agree without a login flow.
DEMO_ORG_ID = UUID("00000000-0000-0000-0000-000000000001")
DEMO_USER_ID = uuid4()
