"""Enterprise governance: budgets, policies, quotas and enforcement.

## Enforcement happens pre-flight, not post-hoc

A budget system that only reports overruns after the fact is an accounting
tool, not a control. The value is in the pre-flight check: the SDK calls
`evaluate_request` *before* dispatching to the provider, and the policy engine
returns an action. That path has a hard latency budget of 5ms p99 — anything
slower and teams disable it, at which point the platform governs nothing.

Meeting 5ms means the decision must be made against Redis-cached counters, not
Postgres. The counters are eventually consistent with the ledger (a few seconds
behind), which is a deliberate trade: **we accept a small over-spend at the
boundary in exchange for never being in the request's critical path for a
database round trip.** A budget that blocks at $10,003 instead of exactly
$10,000 is operationally fine; a 40ms tax on every inference call is not.

## Fail-open, by default

If the policy service is unreachable, `evaluate_request` returns ALLOW. An
availability failure in the cost-governance layer must never become an
availability failure in the customer's product. Tenants who need
fail-closed semantics for a specific scope can set it per policy, and the
choice is surfaced explicitly in the UI rather than buried in config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from app.domain.enums import (
    BudgetPeriod,
    BudgetScope,
    EnforcementAction,
    Role,
    Severity,
)
from app.domain.money import ZERO, quantize_cost, safe_div

#: Role -> permission set. Flattened rather than hierarchical: an explicit
#: matrix is auditable at a glance, which is what a SOC 2 reviewer asks for.
#: Hierarchy is applied at assignment time (granting ADMIN implies the lower
#: sets), not at check time, so a permission can be revoked from a high role
#: without unpicking an inheritance chain.
ROLE_PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.VIEWER: frozenset({"dashboard:read", "usage:read", "forecast:read"}),
    Role.DEVELOPER: frozenset(
        {
            "dashboard:read",
            "usage:read",
            "forecast:read",
            "prompt:read",
            "prompt:write",
            "simulation:run",
            "recommendation:read",
        }
    ),
    Role.ANALYST: frozenset(
        {
            "dashboard:read",
            "usage:read",
            "forecast:read",
            "prompt:read",
            "simulation:run",
            "recommendation:read",
            "report:export",
            "anomaly:read",
        }
    ),
    Role.FINOPS: frozenset(
        {
            "dashboard:read",
            "usage:read",
            "forecast:read",
            "prompt:read",
            "simulation:run",
            "recommendation:read",
            "recommendation:write",
            "report:export",
            "anomaly:read",
            "budget:read",
            "budget:write",
            "chargeback:read",
            "chargeback:write",
            "policy:read",
        }
    ),
    Role.APPROVER: frozenset(
        {
            "dashboard:read",
            "usage:read",
            "forecast:read",
            "recommendation:read",
            "approval:decide",
            "budget:read",
            "policy:read",
            "anomaly:read",
        }
    ),
    Role.ADMIN: frozenset(
        {
            "dashboard:read",
            "usage:read",
            "forecast:read",
            "prompt:read",
            "prompt:write",
            "simulation:run",
            "recommendation:read",
            "recommendation:write",
            "report:export",
            "anomaly:read",
            "anomaly:write",
            "budget:read",
            "budget:write",
            "chargeback:read",
            "chargeback:write",
            "policy:read",
            "policy:write",
            "user:read",
            "user:write",
            "approval:decide",
            "audit:read",
        }
    ),
    Role.OWNER: frozenset({"*"}),
}


def has_permission(role: Role, permission: str) -> bool:
    granted = ROLE_PERMISSIONS.get(role, frozenset())
    return "*" in granted or permission in granted


@dataclass
class Budget:
    scope: BudgetScope
    scope_id: str
    amount: Decimal
    period: BudgetPeriod
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    currency: str = "USD"
    #: Fractions of the budget at which to notify, ascending.
    alert_thresholds: tuple[Decimal, ...] = (Decimal("0.5"), Decimal("0.8"), Decimal("0.95"))
    #: What happens at 100%. Defaults to WARN — a platform that starts blocking
    #: production traffic on day one gets uninstalled on day two. Teams opt in
    #: to harder enforcement once they trust the numbers.
    action_at_limit: EnforcementAction = EnforcementAction.WARN
    #: Optional harder action once spend exceeds the limit by this multiple.
    hard_stop_multiplier: Optional[Decimal] = None
    #: Allow the period's unspent remainder to carry forward. Used by teams
    #: with lumpy batch workloads, where a strict monthly cap forces artificial
    #: work-shaping at month boundaries.
    rollover: bool = False
    enabled: bool = True

    def period_bounds(self, at: date | None = None) -> tuple[date, date]:
        today = at or datetime.now(timezone.utc).date()
        if self.period is BudgetPeriod.DAILY:
            return today, today
        if self.period is BudgetPeriod.WEEKLY:
            start = today - timedelta(days=today.weekday())
            return start, start + timedelta(days=6)
        if self.period is BudgetPeriod.MONTHLY:
            start = today.replace(day=1)
            next_month = (start + timedelta(days=32)).replace(day=1)
            return start, next_month - timedelta(days=1)
        if self.period is BudgetPeriod.QUARTERLY:
            quarter = (today.month - 1) // 3
            start = date(today.year, quarter * 3 + 1, 1)
            end_month = start.month + 3
            end = (
                date(start.year + 1, 1, 1) if end_month > 12 else date(start.year, end_month, 1)
            ) - timedelta(days=1)
            return start, end
        return date(today.year, 1, 1), date(today.year, 12, 31)


@dataclass
class BudgetStatus:
    budget: Budget
    spent: Decimal
    period_start: date
    period_end: date
    forecast_period_spend: Optional[Decimal] = None

    @property
    def utilisation(self) -> Decimal:
        return safe_div(self.spent, self.budget.amount)

    @property
    def remaining(self) -> Decimal:
        return self.budget.amount - self.spent

    @property
    def is_exceeded(self) -> bool:
        return self.spent >= self.budget.amount

    @property
    def crossed_thresholds(self) -> list[Decimal]:
        return [t for t in self.budget.alert_thresholds if self.utilisation >= t]

    @property
    def severity(self) -> Severity:
        util = self.utilisation
        if util >= Decimal("1.0"):
            return Severity.CRITICAL
        if util >= Decimal("0.95"):
            return Severity.HIGH
        if util >= Decimal("0.8"):
            return Severity.MEDIUM
        if util >= Decimal("0.5"):
            return Severity.LOW
        return Severity.INFO

    @property
    def projected_to_exceed(self) -> bool:
        """Forecast-based early warning.

        Notifying at 95% utilisation on the 8th of the month is far more useful
        than notifying at 100% on the 30th, when nothing can be done about it.
        """
        if self.forecast_period_spend is None:
            return False
        return self.forecast_period_spend > self.budget.amount


@dataclass
class Policy:
    """A governance rule evaluated pre-flight.

    Conditions are a flat mapping rather than an expression language: an
    expression DSL is more powerful but becomes an un-reviewable surface in a
    compliance audit, and every field here maps to something a policy owner can
    read without training.
    """

    name: str
    action: EnforcementAction
    id: UUID = field(default_factory=uuid4)
    scope: BudgetScope = BudgetScope.ORGANIZATION
    scope_id: Optional[str] = None
    max_cost_per_request: Optional[Decimal] = None
    max_tokens_per_request: Optional[int] = None
    max_requests_per_minute: Optional[int] = None
    max_context_tokens: Optional[int] = None
    allowed_providers: Optional[set[str]] = None
    blocked_models: set[str] = field(default_factory=set)
    require_approval_above: Optional[Decimal] = None
    #: If the policy service cannot evaluate, block instead of allowing.
    #: Off by default; see module docstring.
    fail_closed: bool = False
    enabled: bool = True
    priority: int = 100

    def evaluate(self, request: RequestIntent) -> Optional[PolicyViolation]:
        if not self.enabled:
            return None
        if self.max_cost_per_request is not None and request.estimated_cost > self.max_cost_per_request:
            return PolicyViolation(
                policy=self,
                reason=(
                    f"Estimated cost {quantize_cost(request.estimated_cost)} exceeds the "
                    f"per-request ceiling of {self.max_cost_per_request}."
                ),
            )
        if self.max_tokens_per_request is not None and request.estimated_tokens > self.max_tokens_per_request:
            return PolicyViolation(
                policy=self,
                reason=(
                    f"{request.estimated_tokens:,} tokens exceeds the per-request limit of "
                    f"{self.max_tokens_per_request:,}."
                ),
            )
        if self.max_context_tokens is not None and request.context_tokens > self.max_context_tokens:
            return PolicyViolation(
                policy=self,
                reason=(
                    f"Context of {request.context_tokens:,} tokens exceeds the "
                    f"{self.max_context_tokens:,} limit."
                ),
            )
        if self.allowed_providers and request.provider not in self.allowed_providers:
            return PolicyViolation(
                policy=self,
                reason=f"Provider '{request.provider}' is not on the approved vendor list.",
            )
        if request.model in self.blocked_models:
            return PolicyViolation(policy=self, reason=f"Model '{request.model}' is blocked by policy.")
        if self.require_approval_above is not None and request.estimated_cost > self.require_approval_above:
            return PolicyViolation(
                policy=self,
                reason=(
                    f"Estimated cost {quantize_cost(request.estimated_cost)} requires approval "
                    f"(threshold {self.require_approval_above})."
                ),
                override_action=EnforcementAction.REQUIRE_APPROVAL,
            )
        return None


@dataclass
class RequestIntent:
    """What the caller is about to do, as supplied by the SDK pre-flight."""

    provider: str
    model: str
    estimated_tokens: int
    estimated_cost: Decimal
    context_tokens: int = 0
    department_id: Optional[str] = None
    team_id: Optional[str] = None
    user_id: Optional[str] = None
    feature: Optional[str] = None
    project: Optional[str] = None
    environment: str = "production"


@dataclass
class PolicyViolation:
    policy: Policy
    reason: str
    override_action: Optional[EnforcementAction] = None

    @property
    def action(self) -> EnforcementAction:
        return self.override_action or self.policy.action


@dataclass
class EnforcementDecision:
    action: EnforcementAction
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    violated_policies: list[str] = field(default_factory=list)
    #: Populated when the action is DOWNGRADE_MODEL — the caller should retry
    #: against this model instead of failing.
    suggested_model: Optional[str] = None
    budget_status: Optional[BudgetStatus] = None
    evaluated_in_ms: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "action": str(self.action),
            "allowed": self.allowed,
            "reasons": self.reasons,
            "violated_policies": self.violated_policies,
            "suggested_model": self.suggested_model,
            "evaluated_in_ms": self.evaluated_in_ms,
        }


#: Ordered by strictness so a multi-policy evaluation can take the max.
_ACTION_SEVERITY: dict[EnforcementAction, int] = {
    EnforcementAction.ALLOW: 0,
    EnforcementAction.WARN: 1,
    EnforcementAction.DOWNGRADE_MODEL: 2,
    EnforcementAction.REQUIRE_APPROVAL: 3,
    EnforcementAction.THROTTLE: 4,
    EnforcementAction.BLOCK: 5,
}


class PolicyEngine:
    """Evaluates policies and budgets for a request, pre-flight."""

    def __init__(self, policies: list[Policy] | None = None) -> None:
        # Higher priority first, so the most specific policy is reported first
        # in the reasons list even though the strictest action always wins.
        self._policies = sorted(policies or [], key=lambda p: p.priority, reverse=True)

    #: Which field of the intent each scope is matched against. A table rather
    #: than a branch chain: adding a scope is one line here, and a policy
    #: reviewer can read the whole matching rule at a glance.
    _SCOPE_FIELDS: dict[BudgetScope, str] = {
        BudgetScope.DEPARTMENT: "department_id",
        BudgetScope.TEAM: "team_id",
        BudgetScope.USER: "user_id",
        BudgetScope.FEATURE: "feature",
        BudgetScope.PROJECT: "project",
    }

    def applicable(self, intent: RequestIntent) -> list[Policy]:
        out: list[Policy] = []
        for policy in self._policies:
            if not policy.enabled:
                continue
            # An org-wide policy, or one with no scope id, applies to everything.
            if policy.scope is BudgetScope.ORGANIZATION or policy.scope_id is None:
                out.append(policy)
                continue
            field_name = self._SCOPE_FIELDS.get(policy.scope)
            if field_name and policy.scope_id == getattr(intent, field_name, None):
                out.append(policy)
        return out

    def evaluate_request(
        self,
        intent: RequestIntent,
        *,
        budget_status: Optional[BudgetStatus] = None,
        fallback_model: Optional[str] = None,
    ) -> EnforcementDecision:
        """Return the enforcement decision for one intended request.

        The strictest action across all triggered rules wins. Reasons from
        every triggered rule are returned, not just the deciding one — a
        developer who fixes only the reason they were shown, then trips the
        next one, loses trust in the system fast.
        """
        started = datetime.now(timezone.utc)
        reasons: list[str] = []
        violated: list[str] = []
        action = EnforcementAction.ALLOW

        for policy in self.applicable(intent):
            violation = policy.evaluate(intent)
            if violation is None:
                continue
            violated.append(policy.name)
            reasons.append(violation.reason)
            if _ACTION_SEVERITY[violation.action] > _ACTION_SEVERITY[action]:
                action = violation.action

        if budget_status is not None and budget_status.budget.enabled:
            budget_action = self._budget_action(budget_status)
            if budget_action is not EnforcementAction.ALLOW:
                reasons.append(
                    f"Budget '{budget_status.budget.scope}:{budget_status.budget.scope_id}' at "
                    f"{budget_status.utilisation * 100:.1f}% of "
                    f"{budget_status.budget.amount} for the {budget_status.budget.period} period."
                )
                violated.append(f"budget:{budget_status.budget.id}")
                if _ACTION_SEVERITY[budget_action] > _ACTION_SEVERITY[action]:
                    action = budget_action

        elapsed = (datetime.now(timezone.utc) - started).total_seconds() * 1000
        return EnforcementDecision(
            action=action,
            allowed=action
            not in {EnforcementAction.BLOCK, EnforcementAction.THROTTLE, EnforcementAction.REQUIRE_APPROVAL},
            reasons=reasons,
            violated_policies=violated,
            suggested_model=fallback_model if action is EnforcementAction.DOWNGRADE_MODEL else None,
            budget_status=budget_status,
            evaluated_in_ms=elapsed,
        )

    def _budget_action(self, status: BudgetStatus) -> EnforcementAction:
        budget = status.budget
        if budget.hard_stop_multiplier is not None and status.utilisation >= budget.hard_stop_multiplier:
            return EnforcementAction.BLOCK
        if status.is_exceeded:
            return budget.action_at_limit
        if status.utilisation >= Decimal("0.95"):
            return EnforcementAction.WARN
        return EnforcementAction.ALLOW


@dataclass
class ChargebackLine:
    """One line of a chargeback or showback statement."""

    cost_center: str
    department: Optional[str]
    direct_cost: Decimal
    #: Share of unattributable platform cost allocated to this cost centre.
    allocated_shared_cost: Decimal
    tokens: int
    requests: int

    @property
    def total(self) -> Decimal:
        return quantize_cost(self.direct_cost + self.allocated_shared_cost)


def build_chargeback(
    direct_costs: dict[str, Decimal],
    *,
    shared_cost: Decimal = ZERO,
    usage_weights: Optional[dict[str, Decimal]] = None,
    department_map: Optional[dict[str, str]] = None,
    token_counts: Optional[dict[str, int]] = None,
    request_counts: Optional[dict[str, int]] = None,
) -> list[ChargebackLine]:
    """Allocate direct and shared cost across cost centres.

    Shared cost — the platform's own infrastructure, unattributed traffic,
    committed-use spend that no single team triggered — is allocated
    proportionally to each centre's direct usage. Proportional allocation is
    chosen over equal-split because it is the method finance teams already use
    for shared infrastructure and therefore needs no defending; equal-split
    penalises small teams and reliably starts turf wars.
    """
    weights = usage_weights or direct_costs
    total_weight = sum(weights.values(), ZERO)
    lines: list[ChargebackLine] = []
    for cost_center, direct in direct_costs.items():
        share = safe_div(weights.get(cost_center, ZERO), total_weight)
        lines.append(
            ChargebackLine(
                cost_center=cost_center,
                department=(department_map or {}).get(cost_center),
                direct_cost=quantize_cost(direct),
                allocated_shared_cost=quantize_cost(shared_cost * share),
                tokens=(token_counts or {}).get(cost_center, 0),
                requests=(request_counts or {}).get(cost_center, 0),
            )
        )
    lines.sort(key=lambda line: line.total, reverse=True)
    return lines


@dataclass
class ApprovalRequest:
    """A spend action awaiting human sign-off."""

    requester_id: UUID
    subject: str
    estimated_cost: Decimal
    justification: str
    id: UUID = field(default_factory=uuid4)
    status: str = "pending"
    approver_id: Optional[UUID] = None
    decided_at: Optional[datetime] = None
    decision_note: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    def decide(self, *, approver_id: UUID, approved: bool, note: str | None = None) -> None:
        """Record a decision.

        Self-approval is rejected here rather than only in the API layer, so
        the invariant holds regardless of which caller reaches it — segregation
        of duties is a control an auditor will test directly against the
        domain, not the HTTP surface.
        """
        if approver_id == self.requester_id:
            raise PermissionError("segregation of duties: a requester cannot approve their own request")
        if self.status != "pending":
            raise ValueError(f"request already {self.status}")
        self.status = "approved" if approved else "rejected"
        self.approver_id = approver_id
        self.decided_at = datetime.now(timezone.utc)
        self.decision_note = note
