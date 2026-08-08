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
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from statistics import median
from typing import Callable, Optional, Union

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
    unit: str = "",
    impact_of: Optional[Callable[[Decimal, Decimal], Decimal]] = None,
) -> Optional[Anomaly]:
    """Score the most recent point of a time series against its own history.

    `impact_of(current, expected)` converts the raw deviation into dollars for
    series that are not already denominated in cost (token counts, latency,
    unit price) — defaulting to the raw delta, which is correct when `series`
    is itself a cost series.
    """
    if len(series) < 5:
        return None
    current, history = series[-1], series[:-1]
    score, expected = robust_zscore(current, history)
    # Only upward deviations matter for cost control. A sudden drop is
    # interesting — it usually means an outage — but it is the availability
    # monitor's job, and folding it in here doubles alert volume.
    if score < threshold or current <= expected:
        return None

    raw_impact = impact_of(current, expected) if impact_of else (current - expected)
    impact = quantize_cost(max(raw_impact, ZERO))
    observed_text = _usd(current) if label == "cost" else f"{current}{unit}"
    expected_text = _usd(expected) if label == "cost" else f"{expected}{unit}"
    return Anomaly(
        kind=kind,
        severity=_severity_for(score, impact),
        title=f"{kind.value.replace('_', ' ').title()} on {scope} {scope_key}",
        detail=(
            f"Observed {label} {observed_text} against an expected {expected_text} "
            f"({pct_change(current, expected):.1f}% above baseline, robust z={score:.1f})."
        ),
        scope=scope,
        scope_key=scope_key,
        observed_value=current,
        expected_value=expected,
        deviation_score=score,
        estimated_impact=impact,
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


def detect_token_spikes(
    token_series: dict[str, list[Decimal]],
    cost_series: dict[str, list[Decimal]],
    *,
    threshold: Decimal = DEFAULT_Z_THRESHOLD,
) -> list[Anomaly]:
    """Token-volume spikes per model, ranked by what the extra tokens cost.

    A token spike that costs almost nothing — a swap to a far cheaper model,
    say — is not interesting. Impact is estimated from the model's own
    blended $/token over the observed window, so the finding is ranked by
    dollars, not by how many tokens moved.
    """

    def token_impact(unit_cost: Decimal) -> Callable[[Decimal, Decimal], Decimal]:
        return lambda current, expected: (current - expected) * unit_cost

    findings: list[Anomaly] = []
    for scope_key, series in token_series.items():
        costs = cost_series.get(scope_key, [])
        total_cost = sum(costs, ZERO)
        total_tokens = sum(series, ZERO)
        unit_cost = safe_div(total_cost, total_tokens) if total_tokens else ZERO

        finding = detect_series_anomaly(
            series=series,
            kind=AnomalyKind.TOKEN_SPIKE,
            scope="model",
            scope_key=scope_key,
            threshold=threshold,
            label="tokens",
            unit=" tokens",
            impact_of=token_impact(unit_cost),
        )
        if finding:
            findings.append(finding)
    return findings


def detect_latency_spikes(
    latency_series: dict[str, list[Decimal]],
    *,
    threshold: Decimal = DEFAULT_Z_THRESHOLD,
) -> list[Anomaly]:
    """Elevated average response latency per model.

    Carries no direct dollar impact — a slow provider does not by itself cost
    more — but latency spikes are frequently the leading indicator of the
    retry storms and timeouts that do, so severity is driven by the
    statistical deviation alone.
    """
    findings: list[Anomaly] = []
    for scope_key, series in latency_series.items():
        finding = detect_series_anomaly(
            series=series,
            kind=AnomalyKind.LATENCY_SPIKE,
            scope="model",
            scope_key=scope_key,
            threshold=threshold,
            label="latency",
            unit=" ms",
            impact_of=lambda current, expected: ZERO,
        )
        if finding:
            finding.recommended_action = (
                "Check provider status; consider failover or a stricter request timeout."
            )
            findings.append(finding)
    return findings


def detect_provider_drift(
    cost_series: dict[str, list[Decimal]],
    token_series: dict[str, list[Decimal]],
    *,
    threshold: Decimal = DEFAULT_Z_THRESHOLD,
    min_daily_tokens: Decimal = Decimal("1000"),
) -> list[Anomaly]:
    """Silent provider price changes, isolated from volume changes.

    Total daily spend rises whenever traffic rises, which is expected and not
    a drift signal. Dividing by that day's token volume gives $/1k tokens — a
    figure that should stay flat between rate-card changes — so a jump in it
    means the provider changed price (or effective discount tier), not that
    the workload simply grew.
    """

    def drift_impact(today_tokens: Decimal) -> Callable[[Decimal, Decimal], Decimal]:
        return lambda current, expected: (current - expected) * safe_div(today_tokens, Decimal("1000"))

    findings: list[Anomaly] = []
    for scope_key, costs in cost_series.items():
        tokens = token_series.get(scope_key, [])
        if len(tokens) != len(costs) or not tokens:
            continue
        unit_series = [
            safe_div(c, t) * Decimal("1000") if t >= min_daily_tokens else ZERO for c, t in zip(costs, tokens)
        ]
        if unit_series[-1] == ZERO:
            continue
        finding = detect_series_anomaly(
            series=unit_series,
            kind=AnomalyKind.PROVIDER_DRIFT,
            scope="model",
            scope_key=scope_key,
            threshold=threshold,
            label="unit cost",
            unit="/1k tokens",
            impact_of=drift_impact(tokens[-1]),
        )
        if finding:
            findings.append(finding)
    return findings


def detect_infinite_loops(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown],
    *,
    repeat_threshold: int = 5,
) -> list[Anomaly]:
    """Agent runs stuck resending the identical prompt.

    Distinct from `detect_runaway_agents`, which flags runs that are merely
    deep or expensive — a legitimate long research agent looks the same on
    those two axes. This instead looks for the mechanical signature of a
    stuck loop: the exact same prompt fingerprint recurring within one agent
    run, which only happens when state genuinely is not advancing between
    steps.
    """
    runs: dict[str, dict[str, list[UsageEvent]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        run_id = event.trace.agent_run_id
        if not run_id or not event.prompt_fingerprint:
            continue
        runs[run_id][event.prompt_fingerprint].append(event)

    findings: list[Anomaly] = []
    for run_id, groups in runs.items():
        stuck = max(groups.values(), key=len, default=[])
        if len(stuck) < repeat_threshold:
            continue
        spend = sum((costs[str(e.id)].total for e in stuck if str(e.id) in costs), ZERO)
        findings.append(
            Anomaly(
                kind=AnomalyKind.INFINITE_LOOP,
                severity=_severity_for(Decimal(len(stuck)), spend),
                title=f"Agent run {run_id[:12]} is repeating an identical call",
                detail=(
                    f"The same prompt was sent {len(stuck)} times within one agent run, "
                    f"costing {_usd(spend)} with no apparent change in state between calls."
                ),
                scope="agent_run",
                scope_key=run_id,
                observed_value=Decimal(len(stuck)),
                expected_value=Decimal(repeat_threshold),
                estimated_impact=quantize_cost(spend),
                evidence={"repeated_calls": len(stuck), "distinct_prompts": len(groups)},
                recommended_action=(
                    "Add a repeated-call guard that aborts the loop when consecutive steps "
                    "produce the identical prompt."
                ),
            )
        )
    return findings


def detect_prompt_injection_bursts(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown],
    *,
    filtered_threshold: int = 10,
    min_requests: int = 20,
) -> list[Anomaly]:
    """Concentrated moderation-filtered requests — the adversarial-probing signal.

    Never inspects prompt content, in keeping with the platform's privacy
    boundary: the provider's own content filter already made the call, and
    this only counts how often it fired per scope. A single filtered request
    is normal; a burst against one user or one feature is someone testing
    what gets through.
    """
    by_scope: dict[str, dict[str, Decimal]] = defaultdict(
        lambda: {"total": ZERO, "filtered": ZERO, "cost": ZERO}
    )
    for event in events:
        scope_key = str(event.attribution.user_id or event.attribution.feature or "unattributed")
        entry = by_scope[scope_key]
        entry["total"] += Decimal("1")
        if event.status is RequestStatus.FILTERED:
            entry["filtered"] += Decimal("1")
            cost = costs.get(str(event.id))
            if cost:
                entry["cost"] += cost.total

    findings: list[Anomaly] = []
    for scope_key, entry in by_scope.items():
        if entry["total"] < min_requests or entry["filtered"] < filtered_threshold:
            continue
        rate = safe_div(entry["filtered"], entry["total"])
        findings.append(
            Anomaly(
                kind=AnomalyKind.PROMPT_INJECTION,
                severity=_severity_for(rate * Decimal("10"), entry["cost"]),
                title=f"Repeated content-filtered requests from {scope_key}",
                detail=(
                    f"{int(entry['filtered'])} of {int(entry['total']):,} requests were blocked "
                    f"by the provider's content filter, costing {_usd(entry['cost'])} for "
                    "content that never reached a user. Consistent with adversarial prompt probing."
                ),
                scope="user_or_feature",
                scope_key=scope_key,
                observed_value=entry["filtered"],
                expected_value=Decimal(filtered_threshold),
                estimated_impact=quantize_cost(entry["cost"]),
                evidence={"requests": int(entry["total"]), "filtered": int(entry["filtered"])},
                recommended_action=(
                    "Review the source for this scope; consider rate-limiting or a stricter "
                    "pre-flight policy for repeated filter violations."
                ),
            )
        )
    return findings


def detect_api_abuse(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown],
    *,
    window_seconds: int = 60,
    request_threshold: int = 120,
) -> list[Anomaly]:
    """Abnormal request rate from a single identity — key sharing, scraping, credential stuffing.

    A sliding window over per-user timestamps, not a flat daily count: 120
    requests spread over a day is unremarkable, the same 120 in one minute is
    not, and only the second pattern indicates automated abuse rather than
    heavy legitimate usage.
    """
    by_user: dict[str, list[UsageEvent]] = defaultdict(list)
    for event in events:
        raw_user_id = event.attribution.user_id
        if raw_user_id:
            by_user[str(raw_user_id)].append(event)

    findings: list[Anomaly] = []
    window = timedelta(seconds=window_seconds)
    for user_id, user_events in by_user.items():
        user_events.sort(key=lambda e: e.occurred_at)
        left = 0
        peak = 0
        peak_slice: list[UsageEvent] = []
        for right in range(len(user_events)):
            while user_events[right].occurred_at - user_events[left].occurred_at > window:
                left += 1
            count = right - left + 1
            if count > peak:
                peak = count
                peak_slice = user_events[left : right + 1]
        if peak < request_threshold:
            continue
        spend = sum((costs[str(e.id)].total for e in peak_slice if str(e.id) in costs), ZERO)
        findings.append(
            Anomaly(
                kind=AnomalyKind.API_ABUSE,
                severity=_severity_for(Decimal(peak) / Decimal("10"), spend),
                title=f"Abnormal request rate from user {user_id[:12]}",
                detail=(
                    f"{peak} requests within {window_seconds} seconds, costing {_usd(spend)}. "
                    "Consistent with a shared key, a scraping loop, or credential stuffing."
                ),
                scope="user",
                scope_key=user_id,
                observed_value=Decimal(peak),
                expected_value=Decimal(request_threshold),
                estimated_impact=quantize_cost(spend),
                evidence={"window_seconds": window_seconds, "peak_requests": peak},
                recommended_action=(
                    "Rate-limit this identity and verify the API key has not been shared or leaked."
                ),
            )
        )
    return findings


def detect_rag_misconfiguration(
    events: list[UsageEvent],
    costs: dict[str, CostBreakdown],
    *,
    ratio_threshold: Decimal = Decimal("0.7"),
    min_requests: int = 20,
) -> list[Anomaly]:
    """RAG context dominating the prompt — the top_k/chunk_size-too-high signature.

    When retrieved chunks make up most of what the model reads, on average,
    across a feature, that is usually over-retrieval rather than a genuinely
    context-hungry task: `top_k` or `chunk_size` set generously "to be safe"
    and never tuned down once retrieval quality was confirmed acceptable at a
    smaller value.
    """
    by_feature: dict[str, dict[str, Decimal]] = defaultdict(
        lambda: {"requests": ZERO, "rag_tokens": ZERO, "prompt_tokens": ZERO, "cost": ZERO}
    )
    for event in events:
        if event.trace.rag_chunks <= 0:
            continue
        scope_key = event.attribution.feature or event.attribution.application or "unattributed"
        entry = by_feature[scope_key]
        entry["requests"] += Decimal("1")
        entry["rag_tokens"] += Decimal(event.trace.rag_tokens)
        entry["prompt_tokens"] += Decimal(event.tokens.prompt_side)
        cost = costs.get(str(event.id))
        if cost:
            entry["cost"] += cost.total

    findings: list[Anomaly] = []
    for scope_key, entry in by_feature.items():
        if entry["requests"] < min_requests or entry["prompt_tokens"] == ZERO:
            continue
        ratio = safe_div(entry["rag_tokens"], entry["prompt_tokens"])
        if ratio < ratio_threshold:
            continue
        # Cost attributable to the RAG share beyond the threshold — a
        # conservative "how much of this looks like over-retrieval" estimate,
        # not the full RAG cost, since some retrieval is always necessary.
        excess_ratio = ratio - ratio_threshold
        recoverable = entry["cost"] * safe_div(excess_ratio, ratio) if ratio > ZERO else ZERO
        if recoverable < Decimal("1"):
            continue
        findings.append(
            Anomaly(
                kind=AnomalyKind.RAG_MISCONFIGURATION,
                severity=_severity_for(ratio * Decimal("10"), recoverable),
                title=f"RAG context dominates prompts in '{scope_key}'",
                detail=(
                    f"Retrieved context is {ratio * 100:.0f}% of prompt tokens on average across "
                    f"{int(entry['requests']):,} requests, costing {_usd(entry['cost'])} of which "
                    f"~{_usd(recoverable)} looks recoverable by tuning top_k or chunk_size."
                ),
                scope="feature",
                scope_key=scope_key,
                observed_value=ratio * Decimal("100"),
                expected_value=ratio_threshold * Decimal("100"),
                estimated_impact=quantize_cost(recoverable),
                evidence={"requests": int(entry["requests"]), "avg_rag_ratio": float(ratio)},
                recommended_action=(
                    "Reduce top_k or chunk_size and re-validate answer quality at the smaller value."
                ),
            )
        )
    return findings


def detect_all(
    events: list[UsageEvent],
    costs: Union[dict[str, CostBreakdown], list[CostBreakdown]],
    *,
    daily_cost_series: Optional[dict[str, list[Decimal]]] = None,
    daily_token_series: Optional[dict[str, list[Decimal]]] = None,
    daily_latency_series: Optional[dict[str, list[Decimal]]] = None,
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
    findings.extend(detect_infinite_loops(events, cost_map))
    findings.extend(detect_prompt_injection_bursts(events, cost_map))
    findings.extend(detect_api_abuse(events, cost_map))
    findings.extend(detect_rag_misconfiguration(events, cost_map))

    for scope_key, series in (daily_cost_series or {}).items():
        finding = detect_series_anomaly(
            series=series,
            kind=AnomalyKind.COST_SPIKE,
            scope="daily_cost",
            scope_key=scope_key,
        )
        if finding:
            findings.append(finding)

    if daily_token_series:
        findings.extend(detect_token_spikes(daily_token_series, daily_cost_series or {}))
        if daily_cost_series:
            findings.extend(detect_provider_drift(daily_cost_series, daily_token_series))
    if daily_latency_series:
        findings.extend(detect_latency_spikes(daily_latency_series))

    findings.sort(key=lambda a: a.estimated_impact, reverse=True)
    return findings
