"""Anomaly and waste detection.

## Detector design

Two families, deliberately separated:

**Statistical detectors** (cost/token/latency spikes) run over aggregated time
series and answer "is today unusual for this series?". They use robust
statistics — median and MAD rather than mean and standard deviation — because
cost series contain the very outliers we are hunting, and a mean-based z-score
is dragged toward the anomaly it is supposed to flag. One $50k runaway job
inflates the standard deviation enough to hide the next one. MAD has a 50%
breakdown point: it stays stable until half the observations are anomalous.

**Structural detectors** (runaway agents, retry storms, infinite loops, context
explosion) run over raw event streams and answer "is this pattern pathological
regardless of history?". These need no baseline, which matters because the
expensive failures are often novel — a newly deployed agent with a broken exit
condition has no history to be anomalous against, and by the time it does, it
has burned a month of budget in a weekend.

Every finding carries an estimated dollar impact. An alert that says "token
usage is 3.2 sigma above baseline" gets muted; one that says "$4,200 burned in
40 minutes by agent run X, still running" gets acted on.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from statistics import median
from typing import Optional, Union

from app.domain.enums import AnomalyKind, RequestStatus, Severity
from app.domain.money import ZERO, pct_change, quantize_cost, safe_div, to_decimal
from app.domain.usage import CostBreakdown, UsageEvent

#: Scales MAD to be a consistent estimator of sigma for normal data, so the
#: resulting score is interpretable on the familiar z-score scale.
MAD_TO_SIGMA = Decimal("1.4826")

#: Robust z above which a point is anomalous. 3.5 (vs the textbook 3.0) is
#: tuned to keep the daily alert volume in single digits for a tenant with a
#: few thousand series — an alerting system nobody reads has negative value.
DEFAULT_Z_THRESHOLD = Decimal("3.5")


@dataclass
class Anomaly:
    kind: AnomalyKind
    severity: Severity
    title: str
    detail: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scope: str = "global"
    scope_key: Optional[str] = None
    observed_value: Decimal = ZERO
    expected_value: Decimal = ZERO
    deviation_score: Decimal = ZERO
    #: Dollars attributable to the anomaly — observed minus expected, floored
    #: at zero. This is what goes in the alert subject line.
    estimated_impact: Decimal = ZERO
    evidence: dict[str, object] = field(default_factory=dict)
    recommended_action: Optional[str] = None

    @property
    def is_actionable(self) -> bool:
        return self.severity in {Severity.HIGH, Severity.CRITICAL}


def _usd(value: Decimal) -> str:
    """Format a Decimal as currency for human-readable alert text.

    Necessary because `str(Decimal)` leaks storage precision and scientific
    notation into user-facing copy — an alert reading "0E-10 was billed" or
    "2.1796287302 is recoverable" looks like a bug regardless of being
    numerically correct.
    """
    return f"${value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"


def _severity_for(score: Decimal, impact: Decimal) -> Severity:
    """Severity blends statistical confidence with financial materiality.

    A 6-sigma deviation on a series that spends $3/day is a curiosity, not an
    incident. Gating on impact prevents the long tail of tiny series from
    generating the majority of alerts, which is the failure mode of every
    purely statistical alerting system.
    """
    if impact >= Decimal("5000") or (score >= Decimal("8") and impact >= Decimal("500")):
        return Severity.CRITICAL
    if impact >= Decimal("1000") or (score >= Decimal("5") and impact >= Decimal("100")):
        return Severity.HIGH
    if impact >= Decimal("100") or score >= Decimal("4"):
        return Severity.MEDIUM
    if impact >= Decimal("10"):
        return Severity.LOW
    return Severity.INFO


def robust_zscore(value: Decimal, history: list[Decimal]) -> tuple[Decimal, Decimal]:
    """Return `(score, expected)` using median/MAD.

    When MAD is zero — a perfectly flat series, common for scheduled batch
    workloads that cost the same every night — we fall back to a ratio against
    the median so that a flat $100/day series jumping to $900 still scores.
    Without the fallback, division by zero would make the most predictable
    series undetectable, which is exactly backwards.
    """
    if len(history) < 4:
        return ZERO, value
    centre = to_decimal(median(history))
    deviations = [abs(h - centre) for h in history]
    mad = to_decimal(median(deviations))
    if mad == ZERO:
        if centre == ZERO:
            return ZERO, centre
        ratio = safe_div(abs(value - centre), centre)
        return ratio * Decimal("10"), centre
    return safe_div(abs(value - centre), mad * MAD_TO_SIGMA), centre


def detect_series_anomaly(
    *,
    series: list[Decimal],
    kind: AnomalyKind,
    scope: str,
    scope_key: str,
    threshold: Decimal = DEFAULT_Z_THRESHOLD,
    label: str = "cost",
) -> Optional[Anomaly]:
    """Score the most recent point of a time series against its own history."""
    if len(series) < 5:
        return None
    current, history = series[-1], series[:-1]
    score, expected = robust_zscore(current, history)
    # Only upward deviations matter for cost control. A sudden drop is
    # interesting — it usually means an outage — but it is the availability
    # monitor's job, and folding it in here doubles alert volume.
    if score < threshold or current <= expected:
        return None

    impact = quantize_cost(current - expected)
    return Anomaly(
        kind=kind,
        severity=_severity_for(score, impact if label == "cost" else impact / Decimal("1000")),
        title=f"{kind.value.replace('_', ' ').title()} on {scope} {scope_key}",
        detail=(
            f"Observed {label} {_usd(current) if label == 'cost' else current} against an "
            f"expected {_usd(expected) if label == 'cost' else expected} "
            f"({pct_change(current, expected):.1f}% above baseline, robust z={score:.1f})."
        ),
        scope=scope,
        scope_key=scope_key,
        observed_value=current,
        expected_value=expected,
        deviation_score=score,
        estimated_impact=impact if label == "cost" else ZERO,
        evidence={"history_points": len(history), "threshold": float(threshold)},
        recommended_action="Inspect the driving requests and confirm the workload change was intended.",
    )


def detect_runaway_agents(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown],
    *,
    step_threshold: int = 25,
    cost_threshold: Decimal = Decimal("50"),
) -> list[Anomaly]:
    """Agent loops that are not terminating.

    Two independent signals, either of which fires:
    - depth beyond `step_threshold` (a reasoning loop that should have
      converged), and
    - cumulative cost for a single agent run beyond `cost_threshold`.

    Cost alone is not sufficient because a legitimately expensive research
    agent can exceed it; depth alone is not sufficient because a shallow loop
    with enormous context can burn more. Reporting both lets the responder
    judge quickly.
    """
    runs: dict[str, dict[str, object]] = defaultdict(
        lambda: {"steps": 0, "cost": ZERO, "max_step": 0, "first": None, "last": None}
    )
    for event in events:
        run_id = event.trace.agent_run_id
        if not run_id:
            continue
        entry = runs[run_id]
        entry["steps"] = int(entry["steps"]) + 1  # type: ignore[arg-type]
        cost = costs.get(str(event.id))
        if cost:
            entry["cost"] = to_decimal(entry["cost"]) + cost.total
        entry["max_step"] = max(int(entry["max_step"]), event.trace.agent_step)  # type: ignore[arg-type]
        if entry["first"] is None or event.occurred_at < entry["first"]:  # type: ignore[operator]
            entry["first"] = event.occurred_at
        if entry["last"] is None or event.occurred_at > entry["last"]:  # type: ignore[operator]
            entry["last"] = event.occurred_at

    findings: list[Anomaly] = []
    for run_id, entry in runs.items():
        steps = int(entry["steps"])  # type: ignore[arg-type]
        depth = int(entry["max_step"])  # type: ignore[arg-type]
        spend = to_decimal(entry["cost"])
        if depth < step_threshold and spend < cost_threshold:
            continue
        first, last = entry["first"], entry["last"]
        duration = (last - first).total_seconds() if first and last else 0  # type: ignore[operator]
        findings.append(
            Anomaly(
                kind=AnomalyKind.RUNAWAY_AGENT,
                severity=_severity_for(Decimal(depth) / Decimal("5"), spend),
                title=f"Agent run {run_id[:12]} may not be terminating",
                detail=(
                    f"{steps} model calls at depth {depth} costing {_usd(spend)} "
                    f"over {duration / 60:.1f} minutes."
                ),
                scope="agent_run",
                scope_key=run_id,
                observed_value=Decimal(depth),
                expected_value=Decimal(step_threshold),
                estimated_impact=quantize_cost(spend),
                evidence={"steps": steps, "max_depth": depth, "duration_seconds": duration},
                recommended_action=(
                    "Add a hard step budget and a cost ceiling to the agent loop; "
                    "review the termination condition."
                ),
            )
        )
    return findings


def detect_retry_storms(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown],
    *,
    retry_ratio_threshold: Decimal = Decimal("0.2"),
    min_requests: int = 50,
) -> list[Anomaly]:
    """Excessive retries — spend with a guaranteed zero business return."""
    by_scope: dict[str, dict[str, Decimal]] = defaultdict(
        lambda: {"requests": ZERO, "retries": ZERO, "retry_cost": ZERO}
    )
    for event in events:
        key = f"{event.provider}/{event.model}"
        entry = by_scope[key]
        entry["requests"] += Decimal("1")
        if event.trace.retry_count > 0:
            entry["retries"] += Decimal(event.trace.retry_count)
            cost = costs.get(str(event.id))
            if cost:
                entry["retry_cost"] += cost.total

    findings: list[Anomaly] = []
    for key, entry in by_scope.items():
        if entry["requests"] < min_requests:
            continue
        ratio = safe_div(entry["retries"], entry["requests"])
        if ratio < retry_ratio_threshold:
            continue
        findings.append(
            Anomaly(
                kind=AnomalyKind.RETRY_STORM,
                severity=_severity_for(ratio * Decimal("10"), entry["retry_cost"]),
                title=f"Retry storm against {key}",
                detail=(
                    f"{ratio:.2f} retry attempts per request across "
                    f"{int(entry['requests']):,} requests, costing {_usd(entry['retry_cost'])} "
                    "with no delivered result."
                ),
                scope="model",
                scope_key=key,
                observed_value=ratio * Decimal("100"),
                expected_value=retry_ratio_threshold * Decimal("100"),
                estimated_impact=quantize_cost(entry["retry_cost"]),
                evidence={"requests": int(entry["requests"]), "retries": int(entry["retries"])},
                recommended_action=(
                    "Switch to exponential backoff with jitter, cap attempts at 3, and "
                    "check provider rate-limit headroom before retrying."
                ),
            )
        )
    return findings


def detect_context_explosion(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown],
    *,
    growth_threshold: Decimal = Decimal("500"),
) -> list[Anomaly]:
    """Conversations whose context grows without bound.

    Detected by comparing prompt size at the first and last observed turn of a
    conversation. Unbounded growth means history is being resent in full, and
    the cost curve is quadratic in turn count — the classic chatbot cost bomb.
    """
    convos: dict[str, list[tuple[int, int]]] = defaultdict(list)
    convo_cost: dict[str, Decimal] = defaultdict(lambda: ZERO)
    for event in events:
        cid = event.trace.conversation_id
        if not cid:
            continue
        convos[cid].append((event.trace.conversation_turn, event.tokens.prompt_side))
        cost = costs.get(str(event.id))
        if cost:
            convo_cost[cid] += cost.total

    findings: list[Anomaly] = []
    for cid, turns in convos.items():
        if len(turns) < 4:
            continue
        turns.sort(key=lambda t: t[0])
        first, last = turns[0][1], turns[-1][1]
        if first <= 0:
            continue
        growth = pct_change(Decimal(last), Decimal(first))
        if growth < growth_threshold:
            continue
        findings.append(
            Anomaly(
                kind=AnomalyKind.CONTEXT_EXPLOSION,
                severity=_severity_for(growth / Decimal("100"), convo_cost[cid]),
                title=f"Context growing without bound in conversation {cid[:12]}",
                detail=(
                    f"Prompt grew from {first} to {last} tokens across {len(turns)} turns "
                    f"({growth:.0f}% growth). Cost scales quadratically with turn count."
                ),
                scope="conversation",
                scope_key=cid,
                observed_value=Decimal(last),
                expected_value=Decimal(first),
                estimated_impact=quantize_cost(convo_cost[cid]),
                evidence={"turns": len(turns), "first_prompt": first, "last_prompt": last},
                recommended_action=(
                    "Apply rolling summarisation beyond N turns and pin the system prompt "
                    "to a provider prompt cache."
                ),
            )
        )
    return findings


def detect_duplicate_requests(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown],
    *,
    min_duplicates: int = 5,
) -> list[Anomaly]:
    """Identical prompts re-sent to the provider — the cache-miss signal.

    Keyed on the SDK-supplied prompt fingerprint (a hash of the normalised
    prompt), so the platform never needs the prompt text itself. That is a
    deliberate privacy property: fingerprints let us prove cache value without
    ingesting customer content that would drag the platform into the tenant's
    data-residency and PII scope.

    Findings are rolled up **per feature**, matching the cache advisor. A FAQ
    endpoint serving twelve questions a few hundred times each is one problem
    with one fix; emitting twelve near-identical rows buries the genuinely
    distinct findings underneath them and trains people to skim past the list.
    """
    scopes: dict[str, dict[str, list[UsageEvent]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event.prompt_fingerprint and not event.served_from_semantic_cache:
            scope = event.attribution.feature or event.attribution.application or "unattributed"
            scopes[scope][event.prompt_fingerprint].append(event)

    findings: list[Anomaly] = []
    for scope, groups in scopes.items():
        repeated = {fp: group for fp, group in groups.items() if len(group) >= min_duplicates}
        if not repeated:
            continue

        recoverable = ZERO
        spend = ZERO
        duplicate_calls = 0
        for group in repeated.values():
            group_spend = sum((costs[str(e.id)].total for e in group if str(e.id) in costs), ZERO)
            spend += group_spend
            # The first call of each distinct prompt is unavoidable; the rest
            # are cacheable.
            recoverable += group_spend * safe_div(Decimal(len(group) - 1), Decimal(len(group)))
            duplicate_calls += len(group) - 1

        if recoverable < Decimal("1"):
            continue

        top = sorted(repeated.items(), key=lambda kv: len(kv[1]), reverse=True)[:3]
        findings.append(
            Anomaly(
                kind=AnomalyKind.COST_SPIKE,
                severity=_severity_for(Decimal(duplicate_calls) / Decimal("100"), recoverable),
                title=f"{duplicate_calls:,} cacheable requests bypassed cache in '{scope}'",
                detail=(
                    f"{len(repeated)} distinct prompts were re-sent to the provider "
                    f"{duplicate_calls:,} times beyond their first call, costing {_usd(spend)} "
                    f"of which {_usd(recoverable)} is recoverable. Most repeated: "
                    + ", ".join(f"{fp[:12]} x{len(group)}" for fp, group in top)
                    + "."
                ),
                scope="feature",
                scope_key=scope,
                observed_value=Decimal(duplicate_calls),
                expected_value=Decimal(len(repeated)),
                estimated_impact=quantize_cost(recoverable),
                evidence={
                    "distinct_prompts": len(repeated),
                    "duplicate_calls": duplicate_calls,
                    "total_cost": float(quantize_cost(spend)),
                },
                recommended_action=(
                    "Enable the response cache for this feature with a TTL matched "
                    "to how fast the underlying data changes."
                ),
            )
        )
    return findings


def detect_error_bursts(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown],
    *,
    error_rate_threshold: Decimal = Decimal("0.15"),
    min_requests: int = 30,
) -> list[Anomaly]:
    """Elevated failure rates per provider — often the first sign of an outage."""
    by_provider: dict[str, dict[str, Decimal]] = defaultdict(
        lambda: {"total": ZERO, "errors": ZERO, "cost": ZERO}
    )
    for event in events:
        entry = by_provider[str(event.provider)]
        entry["total"] += Decimal("1")
        if event.status is not RequestStatus.SUCCESS:
            entry["errors"] += Decimal("1")
            cost = costs.get(str(event.id))
            if cost:
                entry["cost"] += cost.total

    findings: list[Anomaly] = []
    for provider, entry in by_provider.items():
        if entry["total"] < min_requests:
            continue
        rate = safe_div(entry["errors"], entry["total"])
        if rate < error_rate_threshold:
            continue
        findings.append(
            Anomaly(
                kind=AnomalyKind.ERROR_BURST,
                severity=_severity_for(rate * Decimal("10"), entry["cost"]),
                title=f"Elevated error rate on {provider}",
                detail=(
                    f"{rate * 100:.1f}% of {int(entry['total']):,} requests failed. "
                    f"{_usd(entry['cost'])} was billed for work that returned nothing."
                ),
                scope="provider",
                scope_key=provider,
                observed_value=rate * Decimal("100"),
                expected_value=error_rate_threshold * Decimal("100"),
                estimated_impact=quantize_cost(entry["cost"]),
                evidence={"requests": int(entry["total"]), "errors": int(entry["errors"])},
                recommended_action="Check provider status and fail over to the configured secondary.",
            )
        )
    return findings


def detect_all(
    events: list[UsageEvent],
    costs: Union[dict[str, CostBreakdown], list[CostBreakdown]],
    *,
    daily_cost_series: Optional[dict[str, list[Decimal]]] = None,
) -> list[Anomaly]:
    """Run the full detector suite and return findings ranked by impact.

    Ranking by dollars rather than by statistical score is the whole point:
    the responder's first item should be the most expensive one, not the most
    statistically surprising one.
    """
    cost_map = costs if isinstance(costs, dict) else {str(c.event_id): c for c in costs}
    findings: list[Anomaly] = []
    findings.extend(detect_runaway_agents(events, cost_map))
    findings.extend(detect_retry_storms(events, cost_map))
    findings.extend(detect_context_explosion(events, cost_map))
    findings.extend(detect_duplicate_requests(events, cost_map))
    findings.extend(detect_error_bursts(events, cost_map))

    for scope_key, series in (daily_cost_series or {}).items():
        finding = detect_series_anomaly(
            series=series,
            kind=AnomalyKind.COST_SPIKE,
            scope="daily_cost",
            scope_key=scope_key,
        )
        if finding:
            findings.append(finding)

    findings.sort(key=lambda a: a.estimated_impact, reverse=True)
    return findings
