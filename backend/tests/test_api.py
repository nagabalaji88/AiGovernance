"""API contract tests against the running application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_store
from app.demo import seed, seed_governance
from app.main import app
from app.store import AnalyticsStore

API = "/api/v1"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Isolated store per test module, seeded with a short synthetic window."""
    store = AnalyticsStore()
    seed(store, days=45)
    seed_governance(store)
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestSystem:
    def test_health_has_no_dependencies(self, client: TestClient) -> None:
        """A liveness probe that checks the DB turns an incident into an outage."""
        assert client.get("/health").json() == {"status": "ok"}

    def test_readiness_reports_dependency_checks(self, client: TestClient) -> None:
        body = client.get("/ready").json()
        assert body["status"] == "ready"
        assert "checks" in body

    def test_metrics_are_exposed_in_prometheus_format(self, client: TestClient) -> None:
        text = client.get("/metrics").text
        assert "http_requests_total" in text
        assert "aicost_spend_usd_total" in text

    def test_every_response_carries_a_request_id(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.headers["X-Request-ID"]
        assert "X-Response-Time-Ms" in response.headers

    def test_openapi_schema_is_generated(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"]
        assert f"{API}/analytics/summary" in schema["paths"]


class TestAnalytics:
    def test_summary_returns_reconciling_totals(self, client: TestClient) -> None:
        body = client.get(f"{API}/analytics/summary?days=30").json()
        assert float(body["total_cost"]) > 0
        assert body["request_count"] > 0
        assert 0 <= float(body["attribution_coverage_pct"]) <= 100
        assert float(body["waste_ratio_pct"]) >= 0

    def test_breakdown_shares_sum_to_one_hundred(self, client: TestClient) -> None:
        rows = client.get(f"{API}/analytics/breakdown?dimension=model&days=30").json()
        assert rows
        assert abs(sum(float(r["share_pct"]) for r in rows) - 100) < 0.01

    def test_breakdown_is_ranked_by_cost(self, client: TestClient) -> None:
        rows = client.get(f"{API}/analytics/breakdown?dimension=team&days=30").json()
        costs = [float(r["cost"]) for r in rows]
        assert costs == sorted(costs, reverse=True)

    def test_unsupported_dimension_lists_the_valid_ones(self, client: TestClient) -> None:
        response = client.get(f"{API}/analytics/breakdown?dimension=nonsense")
        assert response.status_code == 422
        assert "Supported:" in response.json()["detail"]

    def test_timeseries_is_dense(self, client: TestClient) -> None:
        """Sparse series make charts lie and break the forecaster's spacing."""
        rows = client.get(f"{API}/analytics/timeseries?days=30").json()
        assert len(rows) == 31
        dates = [r["at"] for r in rows]
        assert dates == sorted(dates)

    def test_token_analytics_decomposes_the_prompt(self, client: TestClient) -> None:
        body = client.get(f"{API}/analytics/tokens?days=30").json()
        assert body["prompt_p50"] <= body["prompt_p95"] <= body["prompt_p99"]
        composition = body["composition"]
        assert abs(sum(composition.values()) - 100) < 0.5
        assert composition["system_prompt"] > 0

    def test_heatmap_shape_is_consistent(self, client: TestClient) -> None:
        body = client.get(f"{API}/analytics/heatmap?days=30").json()
        assert len(body["values"]) == len(body["rows"])
        assert all(len(row) == len(body["columns"]) for row in body["values"])

    def test_cost_flow_returns_sankey_links(self, client: TestClient) -> None:
        links = client.get(f"{API}/analytics/flow?days=30").json()
        assert links
        assert {"source", "target", "value"} == set(links[0])
        assert all(link["value"] > 0 for link in links)


class TestForecast:
    def test_returns_requested_horizon_with_intervals(self, client: TestClient) -> None:
        body = client.get(f"{API}/forecast?horizon_days=30&history_days=45").json()
        assert len(body["points"]) == 30
        for point in body["points"]:
            assert float(point["lower"]) <= float(point["value"]) <= float(point["upper"])

    def test_horizon_totals_are_monotonic(self, client: TestClient) -> None:
        body = client.get(f"{API}/forecast?horizon_days=90&history_days=45").json()
        totals = body["horizon_totals"]
        assert float(totals["7d"]) <= float(totals["30d"]) <= float(totals["90d"])


class TestOptimization:
    def test_recommendations_are_ranked_by_priority(self, client: TestClient) -> None:
        rows = client.get(f"{API}/optimize/recommendations?days=30").json()
        assert rows
        scores = [float(r["priority_score"]) for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_every_recommendation_is_actionable(self, client: TestClient) -> None:
        for row in client.get(f"{API}/optimize/recommendations?days=30").json():
            assert row["implementation_steps"]
            assert float(row["estimated_monthly_savings"]) > 0
            assert row["rationale"]

    def test_prompt_analysis_returns_findings_without_echoing_text(
        self, client: TestClient
    ) -> None:
        secret = "Please kindly ensure that account 4111-1111-1111-1111 is verified. " * 6
        body = client.post(
            f"{API}/optimize/prompt",
            json={"text": secret, "monthly_calls": 100_000},
        ).json()
        assert body["findings"]
        assert "4111-1111-1111-1111" not in str(body)
        assert float(body["monthly_savings"]) > 0

    def test_routing_explains_its_choice(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/optimize/route",
            json={
                "expected_input_tokens": 4000,
                "expected_output_tokens": 800,
                "objective": "cheapest",
                "complexity": "simple",
                "baseline_model": "gpt-5",
            },
        ).json()
        assert body["selected"]
        assert body["rationale"]
        assert float(body["savings_vs_baseline"]) >= 0
        assert set(body["selected"]["factors"]) == {
            "cost", "latency", "quality", "reliability"
        }

    def test_routing_respects_a_provider_allow_list(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/optimize/route",
            json={
                "expected_input_tokens": 2000,
                "expected_output_tokens": 400,
                "objective": "cheapest",
                "allowed_providers": ["anthropic"],
            },
        ).json()
        assert body["selected"]["provider"] == "anthropic"

    def test_rag_advisory_does_not_double_count(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/optimize/rag",
            json={
                "chunk_size": 1024, "chunk_overlap": 256, "top_k": 15,
                "monthly_requests": 200_000,
            },
        ).json()
        naive = sum(float(o["monthly_savings"]) for o in body["optimizations"])
        assert float(body["total_monthly_savings"]) <= naive
        assert float(body["safe_savings_no_evaluation_needed"]) <= float(
            body["total_monthly_savings"]
        )

    def test_model_comparison_is_sorted_by_cost(self, client: TestClient) -> None:
        rows = client.get(
            f"{API}/optimize/models/compare"
            "?models=openai/gpt-5,openai/gpt-4.1-mini,anthropic/claude-sonnet-5"
        ).json()
        assert len(rows) == 3
        costs = [r["monthly_cost"] for r in rows]
        assert costs == sorted(costs)

    def test_comparison_rejects_malformed_model_specs(self, client: TestClient) -> None:
        assert client.get(f"{API}/optimize/models/compare?models=garbage").status_code == 422


class TestSimulation:
    def test_scenario_returns_a_costed_verdict(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/simulate",
            json={
                "profile": {
                    "provider": "openai", "model": "gpt-4.1",
                    "monthly_requests": 100_000,
                    "avg_input_tokens": 10_000, "avg_output_tokens": 800,
                    "static_input_tokens": 4_000, "rag_input_tokens": 3_000,
                },
                "levers": [{"type": "enable_prompt_cache", "hit_rate": "0.85"}],
                "scenario_name": "prompt cache",
            },
        ).json()
        scenario = body[0]
        assert scenario["name"] == "prompt cache"
        assert float(scenario["monthly_savings"]) > 0
        assert scenario["recommendation"]

    def test_standard_scenarios_are_offered_by_default(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/simulate",
            json={
                "profile": {
                    "provider": "openai", "model": "gpt-4.1",
                    "monthly_requests": 50_000,
                    "avg_input_tokens": 8_000, "avg_output_tokens": 600,
                    "static_input_tokens": 3_000,
                }
            },
        ).json()
        assert len(body) >= 3
        assert all("recommendation" in scenario for scenario in body)

    def test_switch_model_requires_a_target(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/simulate",
            json={
                "profile": {
                    "provider": "openai", "model": "gpt-4.1",
                    "monthly_requests": 1000,
                    "avg_input_tokens": 1000, "avg_output_tokens": 100,
                },
                "levers": [{"type": "switch_model"}],
            },
        )
        assert response.status_code == 422


class TestAnomalies:
    def test_seeded_pathologies_are_detected(self, client: TestClient) -> None:
        rows = client.get(f"{API}/anomalies?days=7").json()
        kinds = {row["kind"] for row in rows}
        assert "runaway_agent" in kinds

    def test_findings_are_ranked_by_dollar_impact(self, client: TestClient) -> None:
        """The responder's first item must be the most expensive one."""
        rows = client.get(f"{API}/anomalies?days=30").json()
        impacts = [float(r["estimated_impact"]) for r in rows]
        assert impacts == sorted(impacts, reverse=True)

    def test_severity_filter_is_applied(self, client: TestClient) -> None:
        high = client.get(f"{API}/anomalies?days=30&min_severity=high").json()
        assert all(r["severity"] in {"high", "critical"} for r in high)


class TestGovernanceApi:
    def test_budgets_report_utilisation(self, client: TestClient) -> None:
        rows = client.get(f"{API}/governance/budgets").json()
        assert rows
        for row in rows:
            assert row["name"]
            assert float(row["utilisation_pct"]) >= 0
            assert row["period_start"] <= row["period_end"]

    def test_budget_creation_round_trips(self, client: TestClient) -> None:
        created = client.post(
            f"{API}/governance/budgets",
            json={
                "name": "Test budget", "scope": "feature", "scope_id": "faq-answering",
                "amount": "5000", "period": "monthly",
            },
        )
        assert created.status_code == 201
        assert created.json()["name"] == "Test budget"

    def test_preflight_allows_a_normal_request(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/governance/preflight",
            json={
                "provider": "openai", "model": "gpt-4.1",
                "estimated_input_tokens": 2000, "estimated_output_tokens": 400,
            },
        ).json()
        assert body["allowed"]
        assert float(body["estimated_cost"]) > 0

    def test_preflight_blocks_a_request_over_the_context_ceiling(
        self, client: TestClient
    ) -> None:
        """The seeded policy caps context at 400k tokens; 500k must be refused."""
        body = client.post(
            f"{API}/governance/preflight",
            json={
                "provider": "anthropic", "model": "claude-opus-5",
                "estimated_input_tokens": 500_000, "estimated_output_tokens": 10_000,
            },
        ).json()
        assert not body["allowed"]
        assert body["action"] == "block"
        assert any("context" in reason.lower() for reason in body["reasons"])

    def test_preflight_requires_approval_above_the_spend_threshold(
        self, client: TestClient
    ) -> None:
        """The seeded policy requires sign-off above $3 per request."""
        body = client.post(
            f"{API}/governance/preflight",
            json={
                "provider": "anthropic", "model": "claude-opus-5",
                "estimated_input_tokens": 400_000, "estimated_output_tokens": 60_000,
            },
        ).json()
        assert body["action"] == "require_approval"
        assert not body["allowed"]

    def test_preflight_is_fast(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/governance/preflight",
            json={
                "provider": "openai", "model": "gpt-4.1",
                "estimated_input_tokens": 1000, "estimated_output_tokens": 200,
            },
        ).json()
        assert body["evaluated_in_ms"] < 50

    def test_unknown_provider_does_not_block_production_traffic(
        self, client: TestClient
    ) -> None:
        """Never fail a customer's request because our catalog is stale."""
        body = client.post(
            f"{API}/governance/preflight",
            json={
                "provider": "some-new-vendor", "model": "x",
                "estimated_input_tokens": 100, "estimated_output_tokens": 50,
            },
        ).json()
        assert body["allowed"]

    def test_chargeback_shares_sum_to_one_hundred(self, client: TestClient) -> None:
        rows = client.get(f"{API}/governance/chargeback?days=30").json()
        assert rows
        assert abs(sum(float(r["share_pct"]) for r in rows) - 100) < 0.01


class TestIngestion:
    def test_accepts_a_batch(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/ingest/events",
            json={
                "events": [
                    {
                        "provider": "openai", "model": "gpt-4.1",
                        "tokens": {"input": 1000, "output": 200},
                        "idempotency_key": "test-key-1",
                        "attribution": {"feature": "unit-test"},
                    }
                ]
            },
        )
        assert body.status_code == 202
        assert body.json()["accepted"] == 1

    def test_idempotency_key_prevents_double_counting(self, client: TestClient) -> None:
        """A retried event that double-charges is worse than a lost one."""
        payload = {
            "events": [
                {
                    "provider": "openai", "model": "gpt-4.1",
                    "tokens": {"input": 500, "output": 100},
                    "idempotency_key": "duplicate-check",
                }
            ]
        }
        client.post(f"{API}/ingest/events", json=payload)
        second = client.post(f"{API}/ingest/events", json=payload).json()
        assert second["accepted"] == 0
        assert second["duplicates"] == 1

    def test_unknown_model_is_accepted_but_flagged(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/ingest/events",
            json={
                "events": [
                    {
                        "provider": "openai", "model": "gpt-99-unreleased",
                        "tokens": {"input": 1000, "output": 200},
                    }
                ]
            },
        ).json()
        assert body["accepted"] == 1
        assert body["unpriced"] == 1
        assert body["errors"]

    def test_unknown_provider_is_rejected_with_a_reason(self, client: TestClient) -> None:
        body = client.post(
            f"{API}/ingest/events",
            json={
                "events": [
                    {
                        "provider": "not-a-provider", "model": "x",
                        "tokens": {"input": 10, "output": 5},
                    }
                ]
            },
        ).json()
        assert body["rejected"] == 1

    def test_tag_cardinality_is_capped(self, client: TestClient) -> None:
        """Unbounded tags are how a metrics system dies."""
        response = client.post(
            f"{API}/ingest/events",
            json={
                "events": [
                    {
                        "provider": "openai", "model": "gpt-4.1",
                        "tokens": {"input": 10, "output": 5},
                        "attribution": {"tags": {f"k{i}": "v" for i in range(30)}},
                    }
                ]
            },
        )
        assert response.status_code == 422

    def test_negative_token_counts_are_rejected(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/ingest/events",
            json={
                "events": [
                    {
                        "provider": "openai", "model": "gpt-4.1",
                        "tokens": {"input": -100, "output": 5},
                    }
                ]
            },
        )
        assert response.status_code == 422


class TestCatalog:
    def test_lists_every_priced_model(self, client: TestClient) -> None:
        rows = client.get(f"{API}/catalog/models").json()
        assert len(rows) >= 25
        providers = {r["provider"] for r in rows}
        assert {"openai", "anthropic", "google_gemini", "aws_bedrock"} <= providers

    def test_every_entry_carries_pricing_and_capability_metadata(
        self, client: TestClient
    ) -> None:
        for row in client.get(f"{API}/catalog/models").json():
            assert row["context_window"] > 0
            assert 0 <= row["quality_index"] <= 1
            assert "supports_prompt_cache" in row
