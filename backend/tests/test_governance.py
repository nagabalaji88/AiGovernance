"""Budgets, policy enforcement, RBAC and chargeback."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.domain.enums import (
    BudgetPeriod,
    BudgetScope,
    EnforcementAction,
    Role,
    Severity,
)
from app.services.governance import (
    ApprovalRequest,
    Budget,
    BudgetStatus,
    Policy,
    PolicyEngine,
    RequestIntent,
    build_chargeback,
    has_permission,
)


def intent(**kwargs) -> RequestIntent:
    defaults = {
        "provider": "openai",
        "model": "gpt-4.1",
        "estimated_tokens": 5_000,
        "estimated_cost": Decimal("0.05"),
        "context_tokens": 4_000,
    }
    defaults.update(kwargs)
    return RequestIntent(**defaults)  # type: ignore[arg-type]


def budget_status(spent: str, amount: str = "1000", **kwargs) -> BudgetStatus:
    budget = Budget(
        scope=BudgetScope.ORGANIZATION,
        scope_id="org-1",
        amount=Decimal(amount),
        period=BudgetPeriod.MONTHLY,
        **kwargs,
    )
    return BudgetStatus(
        budget=budget,
        spent=Decimal(spent),
        period_start=date(2026, 8, 1),
        period_end=date(2026, 8, 31),
    )


class TestRbac:
    def test_owner_has_every_permission(self) -> None:
        assert has_permission(Role.OWNER, "anything:at:all")

    def test_viewer_cannot_write_budgets(self) -> None:
        assert not has_permission(Role.VIEWER, "budget:write")
        assert has_permission(Role.VIEWER, "dashboard:read")

    def test_finops_manages_budgets_but_not_users(self) -> None:
        assert has_permission(Role.FINOPS, "budget:write")
        assert has_permission(Role.FINOPS, "chargeback:write")
        assert not has_permission(Role.FINOPS, "user:write")

    def test_developer_cannot_read_chargeback(self) -> None:
        """Cost attribution across departments is not developer-visible by default."""
        assert not has_permission(Role.DEVELOPER, "chargeback:read")
        assert has_permission(Role.DEVELOPER, "prompt:write")

    def test_approver_can_decide_but_not_edit_policy(self) -> None:
        assert has_permission(Role.APPROVER, "approval:decide")
        assert not has_permission(Role.APPROVER, "policy:write")


class TestBudgetStatus:
    def test_utilisation_and_remaining(self) -> None:
        status = budget_status("250")
        assert status.utilisation == Decimal("0.25")
        assert status.remaining == Decimal("750")
        assert not status.is_exceeded

    def test_severity_escalates_with_utilisation(self) -> None:
        assert budget_status("100").severity is Severity.INFO
        assert budget_status("550").severity is Severity.LOW
        assert budget_status("850").severity is Severity.MEDIUM
        assert budget_status("960").severity is Severity.HIGH
        assert budget_status("1100").severity is Severity.CRITICAL

    def test_crossed_thresholds_are_reported_cumulatively(self) -> None:
        assert budget_status("850").crossed_thresholds == [Decimal("0.5"), Decimal("0.8")]

    def test_forecast_gives_early_warning_before_the_limit(self) -> None:
        """Warning at 95% on the 8th beats warning at 100% on the 30th."""
        status = budget_status("400")
        status.forecast_period_spend = Decimal("1400")
        assert status.projected_to_exceed
        assert not status.is_exceeded


class TestBudgetPeriods:
    def test_monthly_bounds(self) -> None:
        budget = Budget(
            scope=BudgetScope.ORGANIZATION, scope_id="o", amount=Decimal("1"),
            period=BudgetPeriod.MONTHLY,
        )
        start, end = budget.period_bounds(date(2026, 2, 14))
        assert start == date(2026, 2, 1)
        assert end == date(2026, 2, 28)

    def test_quarterly_bounds_across_a_year_boundary(self) -> None:
        budget = Budget(
            scope=BudgetScope.ORGANIZATION, scope_id="o", amount=Decimal("1"),
            period=BudgetPeriod.QUARTERLY,
        )
        start, end = budget.period_bounds(date(2026, 11, 20))
        assert start == date(2026, 10, 1)
        assert end == date(2026, 12, 31)

    def test_weekly_bounds_start_on_monday(self) -> None:
        budget = Budget(
            scope=BudgetScope.ORGANIZATION, scope_id="o", amount=Decimal("1"),
            period=BudgetPeriod.WEEKLY,
        )
        start, end = budget.period_bounds(date(2026, 8, 5))  # a Wednesday
        assert start.weekday() == 0
        assert (end - start).days == 6


class TestPolicyEngine:
    def test_allows_a_compliant_request(self) -> None:
        engine = PolicyEngine([
            Policy(name="cap", action=EnforcementAction.BLOCK,
                   max_cost_per_request=Decimal("10"))
        ])
        decision = engine.evaluate_request(intent())
        assert decision.allowed
        assert decision.action is EnforcementAction.ALLOW

    def test_blocks_an_oversized_request(self) -> None:
        engine = PolicyEngine([
            Policy(name="cap", action=EnforcementAction.BLOCK,
                   max_cost_per_request=Decimal("0.01"))
        ])
        decision = engine.evaluate_request(intent())
        assert not decision.allowed
        assert decision.action is EnforcementAction.BLOCK
        assert "cap" in decision.violated_policies

    def test_strictest_action_wins_across_policies(self) -> None:
        engine = PolicyEngine([
            Policy(name="warn-only", action=EnforcementAction.WARN,
                   max_tokens_per_request=100),
            Policy(name="hard-block", action=EnforcementAction.BLOCK,
                   max_cost_per_request=Decimal("0.001")),
        ])
        decision = engine.evaluate_request(intent())
        assert decision.action is EnforcementAction.BLOCK

    def test_every_violated_rule_is_reported_not_just_the_deciding_one(self) -> None:
        """Fixing one reason then tripping the next destroys trust in the gate."""
        engine = PolicyEngine([
            Policy(name="tokens", action=EnforcementAction.WARN,
                   max_tokens_per_request=100),
            Policy(name="cost", action=EnforcementAction.BLOCK,
                   max_cost_per_request=Decimal("0.001")),
        ])
        decision = engine.evaluate_request(intent())
        assert len(decision.reasons) == 2
        assert set(decision.violated_policies) == {"tokens", "cost"}

    def test_vendor_allow_list_is_enforced(self) -> None:
        engine = PolicyEngine([
            Policy(name="approved-vendors", action=EnforcementAction.BLOCK,
                   allowed_providers={"anthropic"})
        ])
        assert not engine.evaluate_request(intent(provider="openai")).allowed
        assert engine.evaluate_request(intent(provider="anthropic")).allowed

    def test_blocked_model_is_rejected(self) -> None:
        engine = PolicyEngine([
            Policy(name="deprecations", action=EnforcementAction.BLOCK,
                   blocked_models={"gpt-4.1"})
        ])
        assert not engine.evaluate_request(intent()).allowed

    def test_approval_threshold_yields_require_approval(self) -> None:
        engine = PolicyEngine([
            Policy(name="high-spend", action=EnforcementAction.WARN,
                   require_approval_above=Decimal("0.01"))
        ])
        decision = engine.evaluate_request(intent())
        assert decision.action is EnforcementAction.REQUIRE_APPROVAL
        assert not decision.allowed

    def test_scoped_policy_only_applies_to_its_own_team(self) -> None:
        engine = PolicyEngine([
            Policy(name="team-cap", action=EnforcementAction.BLOCK,
                   scope=BudgetScope.TEAM, scope_id="team-a",
                   max_cost_per_request=Decimal("0.001"))
        ])
        assert not engine.evaluate_request(intent(team_id="team-a")).allowed
        assert engine.evaluate_request(intent(team_id="team-b")).allowed

    def test_disabled_policy_is_inert(self) -> None:
        engine = PolicyEngine([
            Policy(name="off", action=EnforcementAction.BLOCK,
                   max_cost_per_request=Decimal("0.001"), enabled=False)
        ])
        assert engine.evaluate_request(intent()).allowed

    def test_budget_overrun_applies_its_configured_action(self) -> None:
        engine = PolicyEngine([])
        status = budget_status("1200", action_at_limit=EnforcementAction.THROTTLE)
        decision = engine.evaluate_request(intent(), budget_status=status)
        assert decision.action is EnforcementAction.THROTTLE
        assert not decision.allowed

    def test_budget_defaults_to_warn_not_block(self) -> None:
        """A platform that blocks production traffic on day one gets uninstalled."""
        engine = PolicyEngine([])
        decision = engine.evaluate_request(intent(), budget_status=budget_status("1200"))
        assert decision.action is EnforcementAction.WARN
        assert decision.allowed

    def test_hard_stop_multiplier_blocks_severe_overruns(self) -> None:
        engine = PolicyEngine([])
        status = budget_status("1600", hard_stop_multiplier=Decimal("1.5"))
        assert engine.evaluate_request(intent(), budget_status=status).action is (
            EnforcementAction.BLOCK
        )

    def test_evaluation_is_fast_enough_for_the_hot_path(self) -> None:
        """SLO: p99 under 5ms. This runs the full policy set 1,000 times."""
        engine = PolicyEngine([
            Policy(name=f"p{i}", action=EnforcementAction.WARN,
                   max_cost_per_request=Decimal("100"))
            for i in range(20)
        ])
        started = datetime.now(UTC)
        for _ in range(1_000):
            engine.evaluate_request(intent())
        elapsed_ms = (datetime.now(UTC) - started).total_seconds() * 1000
        assert elapsed_ms / 1_000 < 5.0


class TestChargeback:
    def test_direct_costs_are_attributed_unchanged(self) -> None:
        lines = build_chargeback({"CC-1": Decimal("600"), "CC-2": Decimal("400")})
        assert {line.cost_center: line.total for line in lines} == {
            "CC-1": Decimal("600.0000000000"),
            "CC-2": Decimal("400.0000000000"),
        }

    def test_shared_cost_is_allocated_proportionally(self) -> None:
        lines = build_chargeback(
            {"CC-1": Decimal("750"), "CC-2": Decimal("250")},
            shared_cost=Decimal("100"),
        )
        by_centre = {line.cost_center: line for line in lines}
        assert by_centre["CC-1"].allocated_shared_cost == Decimal("75.0000000000")
        assert by_centre["CC-2"].allocated_shared_cost == Decimal("25.0000000000")

    def test_total_allocated_equals_total_spend(self) -> None:
        """Chargeback must reconcile exactly or finance will not use it."""
        lines = build_chargeback(
            {"CC-1": Decimal("333.33"), "CC-2": Decimal("666.67")},
            shared_cost=Decimal("250"),
        )
        assert sum(line.total for line in lines) == Decimal("1250.0000000000")

    def test_lines_are_ranked_by_spend(self) -> None:
        lines = build_chargeback(
            {"small": Decimal("10"), "large": Decimal("900"), "mid": Decimal("100")}
        )
        assert [line.cost_center for line in lines] == ["large", "mid", "small"]

    def test_empty_input_does_not_divide_by_zero(self) -> None:
        assert build_chargeback({}, shared_cost=Decimal("100")) == []


class TestApprovals:
    def test_requester_cannot_approve_their_own_request(self) -> None:
        """Segregation of duties, enforced in the domain not just the API."""
        requester = uuid4()
        request = ApprovalRequest(
            requester_id=requester,
            subject="Enable gpt-5 for bulk processing",
            estimated_cost=Decimal("5000"),
            justification="Quarterly backfill",
        )
        with pytest.raises(PermissionError):
            request.decide(approver_id=requester, approved=True)

    def test_approval_records_the_decision(self) -> None:
        request = ApprovalRequest(
            requester_id=uuid4(), subject="s", estimated_cost=Decimal("1"),
            justification="j",
        )
        approver = uuid4()
        request.decide(approver_id=approver, approved=True, note="ok")
        assert request.status == "approved"
        assert request.approver_id == approver
        assert request.decided_at is not None

    def test_a_decided_request_cannot_be_re_decided(self) -> None:
        request = ApprovalRequest(
            requester_id=uuid4(), subject="s", estimated_cost=Decimal("1"),
            justification="j",
        )
        request.decide(approver_id=uuid4(), approved=False)
        with pytest.raises(ValueError):
            request.decide(approver_id=uuid4(), approved=True)
