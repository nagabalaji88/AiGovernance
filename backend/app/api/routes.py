"""HTTP API surface.

Routers are grouped by bounded context. Each endpoint is thin: validate,
delegate to a service, map the result. Business logic never lives here —
that keeps the engines unit-testable without an HTTP client and lets the same
logic serve the API, the Celery workers and the CLI without duplication.

Handlers below operate against an in-memory analytics store so the service is
runnable end-to-end (`make dev`, `make demo`) without Postgres. The repository
interface is the seam where the SQL implementation drops in — see
`app.db.repository`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api import schemas as s
from app.api.deps import Principal, get_principal, get_store
from app.domain.enums import Provider, RoutingObjective, TaskComplexity
from app.domain.money import ZERO, quantize_cost, safe_div
from app.domain.pricing import default_catalog
from app.services import anomaly as anomaly_service
from app.services import cache_advisor, forecasting, prompt_optimizer, rag_optimizer
from app.services import token_analytics as ta
from app.services.model_router import (
    ModelRouter,
    RoutingConstraints,
    RoutingRequest,
    compare_models,
)
from app.services.recommender import RecommendationEngine
from app.services.simulator import (
    BatchRequests,
    CompressPrompt,
    EnablePromptCache,
    EnableResponseCache,
    Lever,
    ReduceRagContext,
    Simulator,
    SummariseHistory,
    SwitchModel,
    WorkloadProfile,
    standard_scenarios,
)
from app.store import AnalyticsStore

router = APIRouter()

_router_engine = ModelRouter()
_simulator = Simulator()
_recommender = RecommendationEngine()


def _window(days: int) -> tuple[datetime, datetime]:
    end = datetime.now(UTC)
    return end - timedelta(days=days), end


def _provider(value: str) -> Provider:
    try:
        return Provider(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unknown provider '{value}'",
        ) from exc


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

ingest_router = APIRouter(prefix="/ingest", tags=["ingestion"])


@ingest_router.post(
    "/events",
    response_model=s.IngestResult,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch of usage events",
)
async def ingest_events(
    payload: s.IngestBatchIn,
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> s.IngestResult:
    """Accept metered usage from an SDK.

    Returns 202, not 201: events are queued and priced asynchronously in the
    production path. A synchronous 201 would tie the caller's latency to our
    write path, which is the opposite of what an observability sidecar should
    do — instrumentation must never be able to slow down or fail the request it
    is measuring.
    """
    principal.require("usage:write")
    return store.ingest(payload.events, organization_id=principal.organization_id)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

analytics_router = APIRouter(prefix="/analytics", tags=["analytics"])


@analytics_router.get("/summary", response_model=s.CostSummaryOut)
async def cost_summary(
    days: int = Query(default=30, ge=1, le=365),
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> s.CostSummaryOut:
    """Headline cost figures, with period-over-period comparison."""
    principal.require("usage:read")
    start, end = _window(days)
    events = store.events_between(principal.organization_id, start, end)
    costs = store.costs_for(events)
    report = ta.analyse(events, costs, dimensions=("provider",))

    prior_start = start - timedelta(days=days)
    prior_events = store.events_between(principal.organization_id, prior_start, start)
    prior_cost = sum((store.cost_of(e).total for e in prior_events), ZERO)

    total_tokens = report.prompt_distribution.total + report.completion_distribution.total
    infra = sum((c.infrastructure_cost for c in costs.values()), ZERO)
    token_cost = sum((c.token_cost for c in costs.values()), ZERO)

    return s.CostSummaryOut(
        total_cost=quantize_cost(report.total_cost),
        token_cost=quantize_cost(token_cost),
        infrastructure_cost=quantize_cost(infra),
        cache_savings=quantize_cost(report.cache_savings),
        wasted_cost=quantize_cost(report.wasted_cost),
        request_count=len(events),
        total_tokens=total_tokens,
        avg_cost_per_request=quantize_cost(safe_div(report.total_cost, Decimal(len(events) or 1))),
        cost_per_1k_tokens=quantize_cost(
            safe_div(report.total_cost, Decimal(total_tokens or 1)) * Decimal("1000")
        ),
        waste_ratio_pct=report.waste_ratio,
        attribution_coverage_pct=report.attribution_coverage,
        period_over_period_pct=(
            safe_div(report.total_cost - prior_cost, prior_cost) * Decimal("100")
            if prior_cost > ZERO
            else None
        ),
    )


@analytics_router.get("/breakdown", response_model=list[s.DimensionSliceOut])
async def cost_breakdown(
    dimension: str = Query(default="model"),
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=20, ge=1, le=200),
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> list[s.DimensionSliceOut]:
    """Cost split across any supported dimension."""
    principal.require("usage:read")
    if dimension not in ta.DIMENSION_EXTRACTORS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unsupported dimension '{dimension}'. "
            f"Supported: {', '.join(sorted(ta.DIMENSION_EXTRACTORS))}",
        )
    start, end = _window(days)
    events = store.events_between(principal.organization_id, start, end)
    costs = store.costs_for(events)
    report = ta.analyse(events, costs, dimensions=(dimension,))
    slices = report.slices.get(dimension, [])[:limit]
    total = report.total_cost

    return [
        s.DimensionSliceOut(
            key=slice_.key,
            label=store.label_for(principal.organization_id, slice_.key),
            requests=slice_.requests,
            total_tokens=slice_.total_tokens,
            cost=quantize_cost(slice_.cost),
            wasted_cost=quantize_cost(slice_.wasted_cost),
            cache_savings=quantize_cost(slice_.cache_savings),
            avg_cost_per_request=quantize_cost(slice_.avg_cost_per_request),
            avg_latency_ms=slice_.avg_latency_ms,
            share_pct=safe_div(slice_.cost, total) * Decimal("100"),
        )
        for slice_ in slices
    ]


@analytics_router.get("/timeseries", response_model=list[s.TimeSeriesPoint])
async def timeseries(
    days: int = Query(default=30, ge=1, le=365),
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> list[s.TimeSeriesPoint]:
    principal.require("usage:read")
    start, end = _window(days)
    return store.daily_series(principal.organization_id, start, end)


@analytics_router.get("/tokens", response_model=s.TokenAnalyticsOut)
async def token_analytics(
    days: int = Query(default=30, ge=1, le=365),
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> s.TokenAnalyticsOut:
    """Prompt decomposition and context efficiency."""
    principal.require("usage:read")
    start, end = _window(days)
    events = store.events_between(principal.organization_id, start, end)
    costs = store.costs_for(events)

    window_size = 0
    if events:
        card = default_catalog.resolve(events[0].provider, events[0].model)
        window_size = card.context_window if card else 0

    report = ta.analyse(events, costs, dimensions=(), window_size=window_size)
    return s.TokenAnalyticsOut(
        prompt_p50=report.prompt_distribution.p50,
        prompt_p95=report.prompt_distribution.p95,
        prompt_p99=report.prompt_distribution.p99,
        completion_p50=report.completion_distribution.p50,
        completion_p95=report.completion_distribution.p95,
        context_p95=report.context_distribution.p95,
        composition={k: float(v) for k, v in report.composition.as_shares().items()},
        static_share_pct=report.context_efficiency.static_share,
        context_growth_per_turn=report.context_efficiency.growth_per_turn,
        window_utilisation_pct=report.context_efficiency.utilisation,
    )


@analytics_router.get("/heatmap", response_model=s.HeatmapOut)
async def heatmap(
    row_dimension: str = Query(default="team"),
    column_dimension: str = Query(default="model"),
    days: int = Query(default=30, ge=1, le=365),
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> s.HeatmapOut:
    principal.require("usage:read")
    start, end = _window(days)
    events = store.events_between(principal.organization_id, start, end)
    costs = store.costs_for(events)
    matrix = ta.token_heatmap(
        events, costs, row_dimension=row_dimension, column_dimension=column_dimension
    )
    raw_rows = sorted(matrix)
    columns = sorted({c for cells in matrix.values() for c in cells})
    # Axis labels must be human-readable for the same reason the Sankey nodes
    # are resolved: a grid of UUIDs conveys nothing. Values stay keyed on the
    # raw ids so lookup is unaffected by the display mapping.
    return s.HeatmapOut(
        rows=[store.label_for(principal.organization_id, r) for r in raw_rows],
        columns=[store.label_for(principal.organization_id, c) for c in columns],
        values=[[float(matrix.get(r, {}).get(c, ZERO)) for c in columns] for r in raw_rows],
    )


@analytics_router.get("/flow", response_model=list[s.SankeyLink])
async def cost_flow(
    days: int = Query(default=30, ge=1, le=365),
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> list[s.SankeyLink]:
    """Sankey links: department -> team -> provider -> model.

    Node keys arrive as `stage:entity_id`; ids are resolved to display names
    here so the chart reads as "Engineering → search-rag → anthropic" rather
    than a chain of UUIDs.
    """
    principal.require("usage:read")
    start, end = _window(days)
    events = store.events_between(principal.organization_id, start, end)
    costs = store.costs_for(events)

    def relabel(node: str) -> str:
        stage, _, entity = node.partition(":")
        return f"{stage}:{store.label_for(principal.organization_id, entity)}"

    return [
        s.SankeyLink(
            source=relabel(str(link["source"])),
            target=relabel(str(link["target"])),
            value=float(link["value"]),  # type: ignore[arg-type]
        )
        for link in ta.cost_flow(events, costs)
    ]


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

forecast_router = APIRouter(prefix="/forecast", tags=["forecasting"])


@forecast_router.get("", response_model=s.ForecastOut)
async def get_forecast(
    horizon_days: int = Query(default=30, ge=1, le=180),
    history_days: int = Query(default=90, ge=14, le=730),
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> s.ForecastOut:
    """Forecast daily spend.

    History is gap-filled to a dense daily series before forecasting —
    exponential smoothing assumes uniform spacing, and a sparse series would
    silently compress the time axis and overstate the trend.

    The current day is dropped before fitting. It is always partial (only the
    hours elapsed so far have been ingested), so including it feeds the model a
    systematically low final observation. That biases the level downward, drags
    the trend negative, and — because the backtest scores its most recent fold
    against that same truncated day — inflates reported MAPE to the point where
    an otherwise sound forecast looks broken.
    """
    principal.require("forecast:read")
    start, end = _window(history_days)
    history = store.dense_daily_costs(principal.organization_id, start, end)
    if history:
        history = history[:-1]
    result = forecasting.forecast_cost(
        history, horizon_days=horizon_days, start_date=end.date() - timedelta(days=1)
    )
    return s.ForecastOut(
        points=[
            s.ForecastPointOut(at=p.at, value=p.value, lower=p.lower, upper=p.upper)
            for p in result.points
        ],
        method=result.method,
        mape=result.mape,
        confidence=result.confidence,
        seasonal=result.seasonal,
        warnings=result.warnings,
        horizon_totals={
            "7d": result.horizon_total(7),
            "30d": result.horizon_total(30),
            "90d": result.horizon_total(min(90, horizon_days)),
        },
    )


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

optimize_router = APIRouter(prefix="/optimize", tags=["optimization"])


@optimize_router.get("/recommendations", response_model=list[s.RecommendationOut])
async def recommendations(
    days: int = Query(default=30, ge=1, le=365),
    include_blocked: bool = Query(default=False),
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> list[s.RecommendationOut]:
    """Every optimization opportunity, ranked by value per unit of effort."""
    principal.require("recommendation:read")
    start, end = _window(days)
    events = store.events_between(principal.organization_id, start, end)
    costs = store.costs_for(events)

    bundle = _recommender.build(
        cache_recommendations=cache_advisor.advise(events, costs),
        anomalies=anomaly_service.detect_all(events, costs),
    )
    items = bundle.recommendations
    if not include_blocked:
        items = [r for r in items if not r.blocked_by_quality]
    return [s.RecommendationOut(**r.as_dict()) for r in items]  # type: ignore[arg-type]


@optimize_router.post("/prompt", response_model=s.PromptAnalysisOut)
async def analyse_prompt(
    payload: s.PromptAnalysisIn,
    principal: Principal = Depends(get_principal),
) -> s.PromptAnalysisOut:
    """Analyse a prompt for removable tokens.

    The submitted text is analysed in-process and never persisted; only derived
    metrics and the fingerprint are returned. See prompt_optimizer.py.
    """
    principal.require("prompt:read")
    analysis = prompt_optimizer.analyse_prompt(
        payload.text,
        exact_token_count=payload.exact_token_count,
        static_prefix_chars=payload.static_prefix_chars,
    )
    suggestion = prompt_optimizer.suggest_compression(
        analysis,
        rate_per_token=payload.rate_per_token,
        monthly_calls=payload.monthly_calls,
    )
    return s.PromptAnalysisOut(
        fingerprint=analysis.fingerprint,
        total_tokens=analysis.total_tokens,
        recoverable_tokens=analysis.recoverable_tokens,
        compression_ratio_pct=analysis.compression_ratio,
        cacheable_ratio_pct=analysis.cacheable_ratio,
        efficiency_score=prompt_optimizer.score_prompt(analysis),
        monthly_savings=Decimal(str(suggestion["monthly_savings"])),
        annual_savings=Decimal(str(suggestion["annual_savings"])),
        quality_risk=str(suggestion["quality_risk"]),
        findings=[s.PromptFindingOut(**f.as_dict()) for f in analysis.findings],  # type: ignore[arg-type]
    )


@optimize_router.post("/route", response_model=s.RoutingOut)
async def route_model(
    payload: s.RoutingIn,
    principal: Principal = Depends(get_principal),
) -> s.RoutingOut:
    """Recommend a model for a workload under the given objective."""
    principal.require("recommendation:read")
    constraints = RoutingConstraints(
        requires_vision=payload.requires_vision,
        requires_tools=payload.requires_tools,
        max_latency_ms=payload.max_latency_ms,
        allowed_providers=(
            {_provider(p) for p in payload.allowed_providers} if payload.allowed_providers else None
        ),
        require_self_hosted=payload.require_self_hosted,
    )
    request = RoutingRequest(
        expected_input_tokens=payload.expected_input_tokens,
        expected_output_tokens=payload.expected_output_tokens,
        objective=RoutingObjective(payload.objective),
        complexity=TaskComplexity(payload.complexity),
        constraints=constraints,
    )
    decision = _router_engine.route(request, baseline_model=payload.baseline_model)

    def to_out(candidate) -> s.RoutingCandidateOut:  # type: ignore[no-untyped-def]
        return s.RoutingCandidateOut(
            provider=str(candidate.provider),
            model=candidate.model,
            estimated_cost=candidate.estimated_cost,
            estimated_latency_ms=candidate.estimated_latency_ms,
            quality=candidate.quality,
            score=candidate.score,
            factors=candidate.factors,
        )

    return s.RoutingOut(
        selected=to_out(decision.selected) if decision.selected else None,
        alternatives=[to_out(c) for c in decision.alternatives],
        baseline=to_out(decision.baseline) if decision.baseline else None,
        savings_vs_baseline=decision.savings_vs_baseline,
        savings_pct=decision.savings_pct,
        quality_delta=decision.quality_delta,
        rationale=decision.rationale,
        rejected_count=len(decision.rejected),
        rejected_sample=dict(list(decision.rejected.items())[:8]),
    )


@optimize_router.post("/rag", response_model=s.RagAdvisoryOut)
async def optimise_rag(
    payload: s.RagConfigIn,
    principal: Principal = Depends(get_principal),
) -> s.RagAdvisoryOut:
    principal.require("recommendation:read")
    config = rag_optimizer.RagConfig(
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        top_k=payload.top_k,
        reranker_enabled=payload.reranker_enabled,
        hybrid_search=payload.hybrid_search,
        embedding_model=payload.embedding_model,
    )
    advisory = rag_optimizer.advise(
        config,
        input_rate_per_token=payload.input_rate_per_token,
        monthly_requests=payload.monthly_requests,
        avg_answer_span_tokens=payload.avg_answer_span_tokens,
    )
    return s.RagAdvisoryOut(
        current_context_tokens_per_request=config.context_tokens,
        overlap_waste_tokens_per_request=config.overlap_waste_tokens,
        total_monthly_savings=advisory.total_monthly_savings,
        safe_savings_no_evaluation_needed=advisory.safe_savings,
        optimizations=[
            s.RagOptimizationOut(
                lever=o.lever,
                current=o.current,
                proposed=o.proposed,
                tokens_saved_per_request=o.tokens_saved_per_request,
                monthly_savings=o.monthly_savings,
                estimated_recall_delta=o.estimated_recall_delta,
                quality_risk=o.quality_risk,
                confidence=o.confidence,
                requires_evaluation=o.requires_evaluation,
                rationale=o.rationale,
            )
            for o in advisory.optimizations
        ],
    )


@optimize_router.get("/models/compare")
async def compare(
    models: str = Query(description="Comma-separated provider/model pairs"),
    input_tokens: int = Query(default=4000, ge=0),
    output_tokens: int = Query(default=800, ge=0),
    monthly_requests: int = Query(default=100_000, ge=0),
    principal: Principal = Depends(get_principal),
) -> list[dict[str, object]]:
    principal.require("usage:read")
    pairs: list[tuple[Provider, str]] = []
    for entry in models.split(","):
        if "/" not in entry:
            continue
        provider_raw, model = entry.split("/", 1)
        pairs.append((_provider(provider_raw.strip()), model.strip()))
    if not pairs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="provide models as 'provider/model' pairs, e.g. openai/gpt-4.1,anthropic/claude-sonnet-5",
        )
    return compare_models(
        default_catalog,
        pairs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        monthly_requests=monthly_requests,
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

simulate_router = APIRouter(prefix="/simulate", tags=["simulation"])

_LEVER_BUILDERS = {
    "compress_prompt": lambda spec: CompressPrompt(
        reduction_pct=spec.reduction_pct or Decimal("20"),
        quality_delta=spec.quality_delta if spec.quality_delta is not None else Decimal("-0.005"),
    ),
    "enable_prompt_cache": lambda spec: EnablePromptCache(
        hit_rate=spec.hit_rate or Decimal("0.85")
    ),
    "enable_response_cache": lambda spec: EnableResponseCache(
        hit_rate=spec.hit_rate or Decimal("0.2")
    ),
    "reduce_rag_context": lambda spec: ReduceRagContext(
        reduction_pct=spec.reduction_pct or Decimal("30"),
        quality_delta=spec.quality_delta if spec.quality_delta is not None else Decimal("-0.01"),
    ),
    "summarise_history": lambda spec: SummariseHistory(
        reduction_pct=spec.reduction_pct or Decimal("60")
    ),
    "batch_requests": lambda spec: BatchRequests(
        eligible_ratio=spec.eligible_ratio or Decimal("0.4")
    ),
}


def _build_lever(spec: s.SimulationLeverIn) -> Lever:
    if spec.type == "switch_model":
        if not spec.provider or not spec.model:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="switch_model requires both 'provider' and 'model'",
            )
        return SwitchModel(
            provider=_provider(spec.provider),
            model=spec.model,
            quality_delta=spec.quality_delta or ZERO,
        )
    builder = _LEVER_BUILDERS.get(spec.type)
    if builder is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unsupported lever '{spec.type}'",
        )
    return builder(spec)


@simulate_router.post("", response_model=list[s.ScenarioOut])
async def simulate(
    payload: s.SimulationIn,
    principal: Principal = Depends(get_principal),
) -> list[s.ScenarioOut]:
    """Price one or more hypothetical changes before making them."""
    principal.require("simulation:run")
    profile = WorkloadProfile(
        provider=_provider(payload.profile.provider),
        model=payload.profile.model,
        monthly_requests=payload.profile.monthly_requests,
        avg_input_tokens=payload.profile.avg_input_tokens,
        avg_output_tokens=payload.profile.avg_output_tokens,
        avg_cached_input_tokens=payload.profile.avg_cached_input_tokens,
        avg_reasoning_tokens=payload.profile.avg_reasoning_tokens,
        static_input_tokens=payload.profile.static_input_tokens,
        rag_input_tokens=payload.profile.rag_input_tokens,
        quality_index=payload.profile.quality_index,
        cacheable_request_ratio=payload.profile.cacheable_request_ratio,
    )

    results = []
    if payload.levers:
        levers = [_build_lever(spec) for spec in payload.levers]
        results.append(_simulator.run(profile, levers, name=payload.scenario_name))
    if payload.include_standard_scenarios or not payload.levers:
        results.extend(
            _simulator.compare(profile, standard_scenarios(profile))
        )
    return [s.ScenarioOut(**r.as_dict()) for r in results]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------

anomaly_router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@anomaly_router.get("", response_model=list[s.AnomalyOut])
async def list_anomalies(
    days: int = Query(default=7, ge=1, le=90),
    min_severity: str = Query(default="low"),
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> list[s.AnomalyOut]:
    principal.require("anomaly:read")
    start, end = _window(days)
    events = store.events_between(principal.organization_id, start, end)
    costs = store.costs_for(events)
    series = store.daily_cost_series_by_model(principal.organization_id, start, end)
    findings = anomaly_service.detect_all(events, costs, daily_cost_series=series)

    order = ["info", "low", "medium", "high", "critical"]
    floor = order.index(min_severity) if min_severity in order else 0
    return [
        s.AnomalyOut(
            id=uuid4(),
            kind=str(f.kind),
            severity=str(f.severity),
            title=f.title,
            detail=f.detail,
            scope=f.scope,
            scope_key=f.scope_key,
            observed_value=f.observed_value,
            expected_value=f.expected_value,
            deviation_score=f.deviation_score,
            estimated_impact=f.estimated_impact,
            evidence=f.evidence,
            recommended_action=f.recommended_action,
            detected_at=f.detected_at,
        )
        for f in findings
        if order.index(str(f.severity)) >= floor
    ]


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

governance_router = APIRouter(prefix="/governance", tags=["governance"])


@governance_router.post("/preflight", response_model=s.PreflightOut)
async def preflight(
    payload: s.PreflightIn,
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> s.PreflightOut:
    """Pre-flight budget and policy check.

    Called by the SDK before dispatching to a provider. Hard latency budget of
    5ms p99 — see governance.py for why this path never touches Postgres.
    """
    principal.require("usage:write")
    return store.evaluate_preflight(payload, organization_id=principal.organization_id)


@governance_router.get("/budgets", response_model=list[s.BudgetStatusOut])
async def list_budgets(
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> list[s.BudgetStatusOut]:
    principal.require("budget:read")
    return store.budget_statuses(principal.organization_id)


@governance_router.post("/budgets", response_model=s.BudgetStatusOut, status_code=201)
async def create_budget(
    payload: s.BudgetIn,
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> s.BudgetStatusOut:
    principal.require("budget:write")
    return store.create_budget(payload, organization_id=principal.organization_id)


@governance_router.get("/chargeback", response_model=list[s.ChargebackLineOut])
async def chargeback(
    days: int = Query(default=30, ge=1, le=365),
    principal: Principal = Depends(get_principal),
    store: AnalyticsStore = Depends(get_store),
) -> list[s.ChargebackLineOut]:
    """Chargeback statement with proportional allocation of shared cost."""
    principal.require("chargeback:read")
    start, end = _window(days)
    return store.chargeback(principal.organization_id, start, end)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

catalog_router = APIRouter(prefix="/catalog", tags=["catalog"])


@catalog_router.get("/models")
async def list_models(
    principal: Principal = Depends(get_principal),
) -> list[dict[str, object]]:
    """The active pricing catalog, as used for all cost resolution."""
    principal.require("usage:read")
    return [
        {
            "provider": str(card.provider),
            "model": card.model,
            "model_type": str(card.model_type),
            "rates_per_million": {str(k): float(v) for k, v in card.rates.items()},
            "context_window": card.context_window,
            "max_output_tokens": card.max_output_tokens,
            "quality_index": float(card.quality_index),
            "latency_ms_per_1k_output": card.latency_ms_per_1k_output,
            "supports_prompt_cache": card.supports_prompt_cache,
            "supports_batch": card.supports_batch,
            "supports_vision": card.supports_vision,
            "effective_from": card.effective_from.isoformat(),
        }
        for card in sorted(default_catalog.all_cards(), key=lambda c: (str(c.provider), c.model))
    ]


router.include_router(ingest_router)
router.include_router(analytics_router)
router.include_router(forecast_router)
router.include_router(optimize_router)
router.include_router(simulate_router)
router.include_router(anomaly_router)
router.include_router(governance_router)
router.include_router(catalog_router)
