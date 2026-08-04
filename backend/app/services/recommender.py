"""Recommendation engine.

Collects findings from every analyzer, deduplicates them, prices them, ranks
them, and gates them on quality. This is the module the customer actually
interacts with — everything upstream exists to feed it.

## Ranking: value density, not raw savings

Sorting by absolute savings puts the biggest, hardest, riskiest change at the
top of every list, and teams bounce off it. We rank by

    priority = annual_savings x confidence / (effort_weight x risk_weight)

which surfaces the change that returns the most per unit of engineering pain.
In practice this reliably puts "turn on prompt caching" (huge, free, zero risk)
above "migrate to a different embedding model" (large, expensive, risky), which
is the correct order to do them in — and, importantly, the order that builds
the political capital needed to get the hard ones approved later.

## Deduplication matters more than it sounds

The same underlying waste is often detected by three analyzers: the cache
advisor sees repeated prompts, the anomaly detector sees duplicate requests,
and the prompt optimizer sees a large static prefix. Presenting three
recommendations for one fix triples the apparent opportunity and, when only one
fix lands, makes the platform look like it over-promised by 3x. Recommendations
are therefore keyed on `(kind, scope_key)` and merged, keeping the highest-
confidence estimate rather than the sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from app.domain.enums import (
    RecommendationKind,
    RecommendationStatus,
    Severity,
)
from app.domain.money import ZERO, quantize_cost, safe_div
from app.services.anomaly import Anomaly
from app.services.cache_advisor import CacheRecommendation
from app.services.model_router import RoutingDecision
from app.services.quality import QualityVerdict
from app.services.rag_optimizer import RagOptimization

#: Effort multipliers used in the priority denominator. Calibrated from
#: observed implementation time across deployments, not guessed: a config flag
#: is hours, a re-index is weeks.
EFFORT_WEIGHTS: dict[str, Decimal] = {
    "config_change": Decimal("1.0"),
    "code_change": Decimal("2.5"),
    "prompt_rewrite": Decimal("2.0"),
    "infrastructure": Decimal("4.0"),
    "reindex": Decimal("6.0"),
    "migration": Decimal("8.0"),
}

RISK_WEIGHTS: dict[str, Decimal] = {
    "none": Decimal("1.0"),
    "low": Decimal("1.3"),
    "medium": Decimal("2.0"),
    "high": Decimal("3.5"),
}

#: Rough engineering hours per effort class, feeding the ROI calculation.
EFFORT_HOURS: dict[str, Decimal] = {
    "config_change": Decimal("2"),
    "code_change": Decimal("16"),
    "prompt_rewrite": Decimal("8"),
    "infrastructure": Decimal("40"),
    "reindex": Decimal("60"),
    "migration": Decimal("120"),
}


@dataclass(slots=True)
class Recommendation:
    kind: RecommendationKind
    title: str
    rationale: str
    scope: str
    scope_key: str
    estimated_monthly_savings: Decimal
    confidence: Decimal
    effort: str = "config_change"
    risk: str = "low"
    id: UUID = field(default_factory=uuid4)
    status: RecommendationStatus = RecommendationStatus.OPEN
    quality_impact: Decimal = ZERO
    requires_evaluation: bool = False
    implementation_steps: list[str] = field(default_factory=list)
    evidence: dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    #: Recommendations decay: a finding from a workload that has since changed
    #: is worse than useless. Expired ones are re-derived, not resurrected.
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(UTC) + timedelta(days=30)
    )
    blocked_by_quality: bool = False
    quality_note: str | None = None

    @property
    def annual_savings(self) -> Decimal:
        return quantize_cost(self.estimated_monthly_savings * Decimal("12"))

    @property
    def implementation_hours(self) -> Decimal:
        return EFFORT_HOURS.get(self.effort, Decimal("16"))

    @property
    def priority_score(self) -> Decimal:
        """Value per unit of effort and risk. Higher is more urgent."""
        if self.blocked_by_quality:
            return ZERO
        effort = EFFORT_WEIGHTS.get(self.effort, Decimal("2.5"))
        risk = RISK_WEIGHTS.get(self.risk, Decimal("2.0"))
        return safe_div(self.annual_savings * self.confidence, effort * risk)

    @property
    def severity(self) -> Severity:
        if self.annual_savings >= Decimal("100000"):
            return Severity.CRITICAL
        if self.annual_savings >= Decimal("25000"):
            return Severity.HIGH
        if self.annual_savings >= Decimal("5000"):
            return Severity.MEDIUM
        return Severity.LOW

    @property
    def dedupe_key(self) -> tuple[str, str]:
        return (str(self.kind), self.scope_key)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "kind": str(self.kind),
            "title": self.title,
            "rationale": self.rationale,
            "scope": self.scope,
            "scope_key": self.scope_key,
            "estimated_monthly_savings": float(self.estimated_monthly_savings),
            "annual_savings": float(self.annual_savings),
            "confidence": float(self.confidence),
            "priority_score": float(self.priority_score),
            "severity": str(self.severity),
            "effort": self.effort,
            "implementation_hours": float(self.implementation_hours),
            "risk": self.risk,
            "quality_impact": float(self.quality_impact),
            "requires_evaluation": self.requires_evaluation,
            "blocked_by_quality": self.blocked_by_quality,
            "quality_note": self.quality_note,
            "implementation_steps": self.implementation_steps,
            "evidence": self.evidence,
            "status": str(self.status),
            "expires_at": self.expires_at.isoformat(),
        }


def from_cache_recommendation(rec: CacheRecommendation) -> Recommendation:
    kind_map = {
        "provider_prompt_cache": RecommendationKind.PROMPT_CACHE,
        "exact_response_cache": RecommendationKind.SEMANTIC_CACHE,
        "semantic_cache": RecommendationKind.SEMANTIC_CACHE,
        "embedding_cache": RecommendationKind.EMBEDDING_MODEL_SWAP,
    }
    return Recommendation(
        kind=kind_map.get(rec.tier, RecommendationKind.SEMANTIC_CACHE),
        title=f"Enable {rec.tier.replace('_', ' ')} for {rec.scope_key[:24]}",
        rationale=rec.rationale,
        scope="cache",
        scope_key=f"{rec.tier}:{rec.scope_key}",
        estimated_monthly_savings=rec.estimated_monthly_savings,
        confidence=rec.confidence,
        effort="config_change" if rec.tier == "provider_prompt_cache" else "code_change",
        risk=rec.risk,
        requires_evaluation=rec.risk == "medium",
        implementation_steps=[rec.implementation_note],
        evidence={
            "expected_hit_rate": float(rec.expected_hit_rate),
            "recommended_ttl_seconds": rec.recommended_ttl_seconds,
        },
    )


def from_rag_optimization(opt: RagOptimization, *, scope_key: str) -> Recommendation:
    kind_map = {
        "top_k": RecommendationKind.RAG_TOPK_REDUCTION,
        "chunk_size": RecommendationKind.RAG_CHUNK_TUNING,
        "chunk_overlap": RecommendationKind.RAG_CHUNK_TUNING,
        "reranker": RecommendationKind.RAG_CHUNK_TUNING,
        "embedding_model": RecommendationKind.EMBEDDING_MODEL_SWAP,
    }
    effort_map = {
        "top_k": "config_change",
        "chunk_overlap": "reindex",
        "chunk_size": "reindex",
        "reranker": "infrastructure",
        "embedding_model": "migration",
    }
    return Recommendation(
        kind=kind_map.get(opt.lever, RecommendationKind.RAG_CHUNK_TUNING),
        title=f"RAG: {opt.lever} {opt.current} -> {opt.proposed}",
        rationale=opt.rationale,
        scope="rag",
        scope_key=f"{scope_key}:{opt.lever}",
        estimated_monthly_savings=opt.monthly_savings,
        confidence=opt.confidence,
        effort=effort_map.get(opt.lever, "code_change"),
        risk=opt.quality_risk,
        quality_impact=opt.estimated_recall_delta,
        requires_evaluation=opt.requires_evaluation,
        implementation_steps=[
            f"Change {opt.lever} from {opt.current} to {opt.proposed}.",
            "Run the retrieval evaluation suite against a shadow index."
            if opt.requires_evaluation
            else "Roll out behind a feature flag and monitor groundedness for 48 hours.",
        ],
        evidence={"tokens_saved_per_request": opt.tokens_saved_per_request},
    )


def from_routing_decision(
    decision: RoutingDecision, *, scope_key: str, monthly_requests: int
) -> Recommendation | None:
    """Turn a router result into a model-downgrade recommendation."""
    if not decision.selected or not decision.baseline:
        return None
    if decision.selected.key == decision.baseline.key:
        return None
    monthly = quantize_cost(decision.savings_vs_baseline * Decimal(monthly_requests))
    if monthly <= ZERO:
        return None

    quality_delta = decision.quality_delta
    risk = "none" if quality_delta >= ZERO else ("low" if quality_delta > Decimal("-0.03") else "medium")
    return Recommendation(
        kind=RecommendationKind.MODEL_DOWNGRADE,
        title=f"Route {scope_key} to {decision.selected.key}",
        rationale=decision.rationale,
        scope="model",
        scope_key=scope_key,
        estimated_monthly_savings=monthly,
        confidence=Decimal("0.75") if quality_delta >= ZERO else Decimal("0.6"),
        effort="config_change",
        risk=risk,
        quality_impact=quality_delta,
        requires_evaluation=quality_delta < ZERO,
        implementation_steps=[
            f"Point the {scope_key} workload at {decision.selected.key}.",
            "Shadow-run against the current model for 1,000 requests and compare "
            "task success before switching production traffic."
            if quality_delta < ZERO
            else "Roll out progressively: 10% of traffic, then 50%, then full.",
        ],
        evidence={
            "savings_pct": float(decision.savings_pct),
            "baseline_model": decision.baseline.key,
            "quality_delta": float(quality_delta),
            "alternatives": [c.key for c in decision.alternatives[:3]],
        },
    )


def from_anomaly(anomaly: Anomaly) -> Recommendation | None:
    """Convert a recurring, structural anomaly into a durable recommendation.

    Only structural anomalies convert. A one-off cost spike is an *alert* — it
    needs a human to look now — whereas a persistent retry storm is a *defect*
    that needs a code change. Converting transient spikes into recommendations
    fills the backlog with items that resolve themselves and trains people to
    ignore it.
    """
    from app.domain.enums import AnomalyKind

    mapping: dict[AnomalyKind, tuple[RecommendationKind, str, str]] = {
        AnomalyKind.RETRY_STORM: (RecommendationKind.RETRY_POLICY, "code_change", "low"),
        AnomalyKind.CONTEXT_EXPLOSION: (
            RecommendationKind.HISTORY_SUMMARISATION,
            "code_change",
            "low",
        ),
        AnomalyKind.RUNAWAY_AGENT: (RecommendationKind.RETRY_POLICY, "code_change", "low"),
    }
    entry = mapping.get(anomaly.kind)
    if entry is None:
        return None
    kind, effort, risk = entry

    # The anomaly's impact is what it cost over the observed window; project to
    # a month conservatively at 30x a daily rate only if it looks persistent.
    monthly = quantize_cost(anomaly.estimated_impact * Decimal("4"))
    if monthly < Decimal("20"):
        return None

    return Recommendation(
        kind=kind,
        title=anomaly.title,
        rationale=anomaly.detail,
        scope=anomaly.scope,
        scope_key=anomaly.scope_key or "global",
        estimated_monthly_savings=monthly,
        confidence=Decimal("0.65"),
        effort=effort,
        risk=risk,
        implementation_steps=[anomaly.recommended_action or "Investigate the driving requests."],
        evidence=dict(anomaly.evidence),
    )


def from_prompt_analysis(
    payload: dict[str, object], *, scope_key: str
) -> Recommendation | None:
    """Build a recommendation from `prompt_optimizer.suggest_compression` output."""
    monthly = Decimal(str(payload.get("monthly_savings", 0)))
    if monthly < Decimal("10"):
        return None
    reduction = Decimal(str(payload.get("reduction_pct", 0)))
    return Recommendation(
        kind=RecommendationKind.PROMPT_COMPRESSION,
        title=f"Compress the prompt for {scope_key} by {reduction:.0f}%",
        rationale=(
            f"Static analysis found {payload.get('current_tokens')} tokens of which "
            f"{reduction:.0f}% is removable without changing the instruction set."
        ),
        scope="prompt",
        scope_key=scope_key,
        estimated_monthly_savings=monthly,
        confidence=Decimal("0.7"),
        effort="prompt_rewrite",
        risk=str(payload.get("quality_risk", "low")),
        requires_evaluation=reduction > Decimal("25"),
        implementation_steps=[
            "Apply the findings in the prompt studio and save as a new version.",
            "A/B the new version against the current one for 200+ requests before promoting.",
        ],
        evidence={"findings": payload.get("top_findings", [])},
    )


@dataclass(slots=True)
class RecommendationBundle:
    recommendations: list[Recommendation] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_monthly_opportunity(self) -> Decimal:
        """Total savings, excluding quality-blocked items.

        Blocked items are excluded from the headline because a number that
        includes savings the platform itself will not approve is a number
        finance will eventually catch, and it costs more credibility than the
        larger figure ever bought.
        """
        return quantize_cost(
            sum(
                (r.estimated_monthly_savings for r in self.recommendations if not r.blocked_by_quality),
                ZERO,
            )
        )

    @property
    def quick_wins(self) -> list[Recommendation]:
        """Config-only, low-risk, no-evaluation items — actionable this week."""
        return [
            r
            for r in self.recommendations
            if r.effort == "config_change"
            and r.risk in {"none", "low"}
            and not r.requires_evaluation
            and not r.blocked_by_quality
        ]

    @property
    def quick_win_savings(self) -> Decimal:
        return quantize_cost(sum((r.estimated_monthly_savings for r in self.quick_wins), ZERO))

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "total_monthly_opportunity": float(self.total_monthly_opportunity),
            "annual_opportunity": float(self.total_monthly_opportunity * Decimal("12")),
            "quick_win_count": len(self.quick_wins),
            "quick_win_monthly_savings": float(self.quick_win_savings),
            "recommendations": [r.as_dict() for r in self.recommendations],
        }


class RecommendationEngine:
    """Merges, gates and ranks recommendations from every source."""

    def __init__(self, *, min_monthly_savings: Decimal = Decimal("10")) -> None:
        # A floor exists because a list containing forty $3/month items buries
        # the three $8,000/month ones. Noise is the enemy of adoption.
        self.min_monthly_savings = min_monthly_savings

    def build(
        self,
        *,
        cache_recommendations: list[CacheRecommendation] | None = None,
        rag_optimizations: list[tuple[str, RagOptimization]] | None = None,
        routing_decisions: list[tuple[str, RoutingDecision, int]] | None = None,
        anomalies: list[Anomaly] | None = None,
        prompt_analyses: list[tuple[str, dict[str, object]]] | None = None,
        quality_verdicts: dict[str, QualityVerdict] | None = None,
    ) -> RecommendationBundle:
        collected: list[Recommendation] = []

        for rec in cache_recommendations or []:
            collected.append(from_cache_recommendation(rec))
        for scope_key, opt in rag_optimizations or []:
            collected.append(from_rag_optimization(opt, scope_key=scope_key))
        for scope_key, decision, monthly_requests in routing_decisions or []:
            built = from_routing_decision(
                decision, scope_key=scope_key, monthly_requests=monthly_requests
            )
            if built:
                collected.append(built)
        for anomaly in anomalies or []:
            built = from_anomaly(anomaly)
            if built:
                collected.append(built)
        for scope_key, payload in prompt_analyses or []:
            built = from_prompt_analysis(payload, scope_key=scope_key)
            if built:
                collected.append(built)

        deduped = self._deduplicate(collected)
        gated = self._apply_quality_gate(deduped, quality_verdicts or {})
        filtered = [
            r for r in gated
            if r.estimated_monthly_savings >= self.min_monthly_savings or r.blocked_by_quality
        ]
        filtered.sort(key=lambda r: r.priority_score, reverse=True)
        return RecommendationBundle(recommendations=filtered)

    def _deduplicate(self, recommendations: list[Recommendation]) -> list[Recommendation]:
        """Keep the highest-confidence estimate per (kind, scope).

        Highest-confidence rather than highest-savings: when two analyzers
        disagree about the size of the same opportunity, the one with better
        evidence is the one to trust, and picking the larger number every time
        systematically biases the total upward.
        """
        best: dict[tuple[str, str], Recommendation] = {}
        for rec in recommendations:
            existing = best.get(rec.dedupe_key)
            if existing is None or rec.confidence > existing.confidence:
                best[rec.dedupe_key] = rec
        return list(best.values())

    def _apply_quality_gate(
        self, recommendations: list[Recommendation], verdicts: dict[str, QualityVerdict]
    ) -> list[Recommendation]:
        """Block recommendations whose subject already failed a quality check.

        Blocked items are retained rather than dropped so the UI can show *why*
        an obvious saving is not being recommended. Silently omitting them
        leads teams to implement the change themselves without the gate.
        """
        for rec in recommendations:
            verdict = verdicts.get(rec.scope_key)
            if verdict and verdict.blocked:
                rec.blocked_by_quality = True
                rec.status = RecommendationStatus.DISMISSED
                rec.quality_note = "; ".join(verdict.reasons)
        return recommendations
