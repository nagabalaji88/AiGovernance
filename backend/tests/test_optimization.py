"""Prompt optimization, model routing, RAG tuning, simulation and quality gates."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.domain.enums import (
    Provider,
    QualityDimension,
    RoutingObjective,
    TaskComplexity,
)
from app.services import prompt_optimizer as po
from app.services import rag_optimizer as ro
from app.services.model_router import (
    ModelRouter,
    RoutingConstraints,
    RoutingRequest,
    classify_complexity,
)
from app.services.quality import (
    QualityBaseline,
    QualityGate,
    QualityMeasurement,
    build_baseline,
    groundedness_score,
    refusal_detected,
)
from app.services.simulator import (
    CompressPrompt,
    EnablePromptCache,
    EnableResponseCache,
    ReduceRagContext,
    Simulator,
    SwitchModel,
    WorkloadProfile,
    roi,
)

# ---------------------------------------------------------------------------
# Prompt optimization
# ---------------------------------------------------------------------------


class TestPromptOptimizer:
    def test_detects_verbatim_duplicate_blocks(self) -> None:
        block = "The customer account balance must be verified against the ledger " * 4
        text = f"Instructions here.\n\n{block}\n\nMore text.\n\n{block}\n\nEnd."
        analysis = po.analyse_prompt(text)
        assert any(f.kind == "duplicate_context" for f in analysis.findings)

    def test_detects_restated_instructions(self) -> None:
        text = (
            "You must always respond using valid JSON format only.\n"
            "Never include any prose outside the JSON structure.\n"
            "You must always respond using valid JSON format only please.\n"
        )
        analysis = po.analyse_prompt(text)
        assert any(f.kind == "repeated_instruction" for f in analysis.findings)

    def test_detects_filler_phrases(self) -> None:
        text = "Please kindly make sure that you carefully review this in order to help."
        analysis = po.analyse_prompt(text)
        assert any(f.kind == "filler_phrase" for f in analysis.findings)

    def test_flags_excess_few_shot_as_low_confidence(self) -> None:
        """Trimming examples can cost accuracy, so it must never look certain."""
        text = "\n".join(f"Example {i}: input -> output" for i in range(1, 9))
        analysis = po.analyse_prompt(text)
        findings = [f for f in analysis.findings if f.kind == "excess_few_shot"]
        assert findings
        assert findings[0].confidence <= Decimal("0.5")

    def test_savings_never_exceed_the_prompt_itself(self) -> None:
        """Overlapping detectors must not claim more tokens than exist."""
        block = "please kindly make sure that you review this in order to proceed. " * 30
        text = f"{block}\n\n{block}\n\n{block}"
        analysis = po.analyse_prompt(text)
        assert analysis.recoverable_tokens <= analysis.total_tokens
        assert analysis.compression_ratio <= Decimal("100")

    def test_clean_prompt_yields_no_material_findings(self) -> None:
        analysis = po.analyse_prompt("Summarise the document in three bullet points.")
        assert analysis.recoverable_tokens == 0

    def test_fingerprint_is_stable_under_cosmetic_edits(self) -> None:
        assert po.fingerprint("Hello   World") == po.fingerprint("hello world")
        assert po.fingerprint("Hello World") != po.fingerprint("Goodbye World")

    def test_score_penalises_waste_and_rewards_cacheable_structure(self) -> None:
        wasteful = po.PromptAnalysis(
            fingerprint="a", total_tokens=1000, recoverable_tokens=400
        )
        clean = po.PromptAnalysis(
            fingerprint="b", total_tokens=1000, recoverable_tokens=0, static_tokens=800
        )
        assert po.score_prompt(clean) > po.score_prompt(wasteful)

    def test_analysis_result_carries_no_prompt_text(self) -> None:
        """Privacy property: only counts, offsets and hashes leave the process."""
        secret = "Customer SSN is 123-45-6789 and must be kept private. " * 5
        payload = po.analyse_prompt(secret).as_dict()
        assert "123-45-6789" not in str(payload)


class TestPromptComparison:
    def test_rejects_promotion_on_insufficient_samples(self) -> None:
        comparison = po.PromptComparison(
            baseline_version="v1", candidate_version="v2",
            baseline_tokens=1000, candidate_tokens=600,
            baseline_quality=Decimal("0.9"), candidate_quality=Decimal("0.9"),
            baseline_cost=Decimal("100"), candidate_cost=Decimal("60"),
            sample_size=15,
        )
        assert "inconclusive" in comparison.verdict()

    def test_quality_regression_vetoes_a_cost_win(self) -> None:
        comparison = po.PromptComparison(
            baseline_version="v1", candidate_version="v2",
            baseline_tokens=1000, candidate_tokens=400,
            baseline_quality=Decimal("0.90"), candidate_quality=Decimal("0.70"),
            baseline_cost=Decimal("100"), candidate_cost=Decimal("40"),
            sample_size=500,
        )
        assert comparison.verdict().startswith("reject")

    def test_promotes_a_cheaper_equal_quality_prompt(self) -> None:
        comparison = po.PromptComparison(
            baseline_version="v1", candidate_version="v2",
            baseline_tokens=1000, candidate_tokens=600,
            baseline_quality=Decimal("0.90"), candidate_quality=Decimal("0.91"),
            baseline_cost=Decimal("100"), candidate_cost=Decimal("60"),
            sample_size=500,
        )
        assert comparison.verdict().startswith("promote")


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------


class TestModelRouter:
    @pytest.fixture
    def router(self) -> ModelRouter:
        return ModelRouter()

    def test_cheapest_objective_picks_a_cheaper_model_than_quality_objective(
        self, router: ModelRouter
    ) -> None:
        base = {
            "expected_input_tokens": 2000,
            "expected_output_tokens": 400,
            "complexity": TaskComplexity.SIMPLE,
        }
        cheap = router.route(RoutingRequest(objective=RoutingObjective.CHEAPEST, **base))
        best = router.route(RoutingRequest(objective=RoutingObjective.HIGHEST_QUALITY, **base))
        assert cheap.selected and best.selected
        assert cheap.selected.estimated_cost <= best.selected.estimated_cost

    def test_quality_floor_blocks_weak_models_on_expert_tasks(
        self, router: ModelRouter
    ) -> None:
        """This is the guard that stops 'cheapest' becoming a quality incident."""
        decision = router.route(
            RoutingRequest(
                expected_input_tokens=2000,
                expected_output_tokens=500,
                objective=RoutingObjective.CHEAPEST,
                complexity=TaskComplexity.EXPERT,
            )
        )
        assert decision.selected
        assert decision.selected.quality >= Decimal("0.94")
        assert any("quality" in reason for reason in decision.rejected.values())

    def test_context_window_is_a_hard_filter(self, router: ModelRouter) -> None:
        decision = router.route(
            RoutingRequest(expected_input_tokens=900_000, expected_output_tokens=1000)
        )
        assert decision.selected
        assert decision.selected.model in {"gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
                                           "gemini-2.5-pro", "gemini-2.5-flash",
                                           "gemini-2.5-flash-lite", "gpt-5", "gpt-5-mini"}

    def test_compliance_allow_list_cannot_be_outvoted_by_cost(
        self, router: ModelRouter
    ) -> None:
        """Compliance is a filter, not a weighted term — by design."""
        decision = router.route(
            RoutingRequest(
                expected_input_tokens=1000,
                expected_output_tokens=200,
                objective=RoutingObjective.CHEAPEST,
                constraints=RoutingConstraints(allowed_providers={Provider.ANTHROPIC}),
            )
        )
        assert decision.selected
        assert decision.selected.provider is Provider.ANTHROPIC

    def test_data_residency_forces_self_hosted(self, router: ModelRouter) -> None:
        decision = router.route(
            RoutingRequest(
                expected_input_tokens=1000,
                expected_output_tokens=200,
                complexity=TaskComplexity.SIMPLE,
                constraints=RoutingConstraints(require_self_hosted=True),
            )
        )
        assert decision.selected
        assert decision.selected.provider in {Provider.VLLM, Provider.OLLAMA}

    def test_vision_requirement_filters_out_text_only_models(
        self, router: ModelRouter
    ) -> None:
        decision = router.route(
            RoutingRequest(
                expected_input_tokens=1000,
                expected_output_tokens=200,
                constraints=RoutingConstraints(requires_vision=True),
            )
        )
        assert decision.selected
        assert "no vision support" in " ".join(decision.rejected.values())

    def test_impossible_constraints_return_an_explanation_not_a_crash(
        self, router: ModelRouter
    ) -> None:
        decision = router.route(
            RoutingRequest(
                expected_input_tokens=50_000_000,
                expected_output_tokens=1000,
            )
        )
        assert decision.selected is None
        assert "constraint" in decision.rationale

    def test_decision_is_explainable(self, router: ModelRouter) -> None:
        decision = router.route(
            RoutingRequest(expected_input_tokens=4000, expected_output_tokens=800),
            baseline_model="gpt-5",
        )
        assert decision.selected
        assert decision.selected.factors.keys() == {"cost", "latency", "quality", "reliability"}
        assert decision.rationale

    def test_measured_quality_overrides_the_catalog_prior(
        self, router: ModelRouter
    ) -> None:
        request = RoutingRequest(
            expected_input_tokens=1000,
            expected_output_tokens=200,
            objective=RoutingObjective.HIGHEST_QUALITY,
            complexity=TaskComplexity.SIMPLE,
            quality_overrides={"groq/llama-4-8b": Decimal("0.999")},
        )
        decision = router.route(request)
        assert decision.selected
        assert decision.selected.key == "groq/llama-4-8b"


class TestComplexityClassifier:
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({"input_tokens": 500}, TaskComplexity.TRIVIAL),
            ({"input_tokens": 12_000}, TaskComplexity.SIMPLE),
            ({"input_tokens": 40_000}, TaskComplexity.MODERATE),
            ({"input_tokens": 40_000, "requires_reasoning": True}, TaskComplexity.COMPLEX),
            (
                {
                    "input_tokens": 40_000,
                    "requires_reasoning": True,
                    "requires_tools": True,
                    "has_structured_output": True,
                },
                TaskComplexity.EXPERT,
            ),
            # Domain sensitivity short-circuits to EXPERT regardless of size:
            # a small legal or medical prompt still needs the best model.
            ({"input_tokens": 100, "domain_sensitive": True}, TaskComplexity.EXPERT),
        ],
    )
    def test_classification(self, kwargs: dict, expected: TaskComplexity) -> None:
        assert classify_complexity(**kwargs) is expected


# ---------------------------------------------------------------------------
# RAG optimization
# ---------------------------------------------------------------------------


class TestRagOptimizer:
    def test_overlap_waste_is_counted_per_retrieved_chunk(self) -> None:
        config = ro.RagConfig(chunk_size=512, chunk_overlap=128, top_k=10)
        assert config.overlap_waste_tokens == 1280
        assert config.context_tokens == 6400

    def test_recommends_trimming_excessive_overlap(self) -> None:
        config = ro.RagConfig(chunk_size=512, chunk_overlap=200, top_k=10)
        result = ro.optimize_overlap(
            config, input_rate_per_token=Decimal("0.000002"), monthly_requests=100_000
        )
        assert result is not None
        assert result.lever == "chunk_overlap"
        assert not result.requires_evaluation
        assert result.monthly_savings > Decimal("0")

    def test_no_overlap_recommendation_when_already_tight(self) -> None:
        config = ro.RagConfig(chunk_size=512, chunk_overlap=50, top_k=10)
        assert ro.optimize_overlap(
            config, input_rate_per_token=Decimal("0.000002"), monthly_requests=100_000
        ) is None

    def test_topk_reduction_without_citation_data_demands_evaluation(self) -> None:
        """Cutting k blind is how a cost programme causes a quality incident."""
        config = ro.RagConfig(top_k=12)
        result = ro.optimize_top_k(
            config, input_rate_per_token=Decimal("0.000002"), monthly_requests=100_000
        )
        assert result is not None
        assert result.requires_evaluation
        assert result.confidence <= Decimal("0.5")

    def test_topk_reduction_with_citation_data_is_confident(self) -> None:
        config = ro.RagConfig(top_k=12)
        citations = {1: Decimal("0.4"), 2: Decimal("0.3"), 3: Decimal("0.2"),
                     4: Decimal("0.05"), 5: Decimal("0.01"), 9: Decimal("0.001")}
        result = ro.optimize_top_k(
            config,
            input_rate_per_token=Decimal("0.000002"),
            monthly_requests=100_000,
            observed_citation_rate=citations,
        )
        assert result is not None
        assert not result.requires_evaluation
        assert int(result.proposed) <= 5

    def test_reranker_improves_quality_while_saving(self) -> None:
        config = ro.RagConfig(top_k=20, chunk_size=512, chunk_overlap=64)
        result = ro.recommend_reranker(
            config, input_rate_per_token=Decimal("0.000002"), monthly_requests=500_000
        )
        assert result is not None
        assert result.estimated_recall_delta > Decimal("0")
        assert result.quality_risk == "none"

    def test_advisory_does_not_triple_count_overlapping_levers(self) -> None:
        """top_k, chunk_size and reranker all act on the same token budget."""
        config = ro.RagConfig(chunk_size=1024, chunk_overlap=256, top_k=15)
        advisory = ro.advise(
            config, input_rate_per_token=Decimal("0.000002"), monthly_requests=200_000
        )
        naive_sum = sum(o.monthly_savings for o in advisory.optimizations)
        assert advisory.total_monthly_savings < naive_sum

    def test_safe_savings_exclude_anything_needing_evaluation(self) -> None:
        config = ro.RagConfig(chunk_size=1024, chunk_overlap=256, top_k=15)
        advisory = ro.advise(
            config, input_rate_per_token=Decimal("0.000002"), monthly_requests=200_000
        )
        assert advisory.safe_savings <= advisory.total_monthly_savings


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class TestSimulator:
    @pytest.fixture
    def profile(self) -> WorkloadProfile:
        return WorkloadProfile(
            provider=Provider.OPENAI,
            model="gpt-4.1",
            monthly_requests=100_000,
            avg_input_tokens=10_000,
            avg_output_tokens=800,
            static_input_tokens=4_000,
            rag_input_tokens=3_000,
            cacheable_request_ratio=Decimal("0.25"),
        )

    def test_levers_compose_multiplicatively_not_additively(
        self, profile: WorkloadProfile
    ) -> None:
        """Summing individual savings overstates combined effect, sometimes >100%."""
        simulator = Simulator()
        compress = simulator.run(profile, [CompressPrompt(reduction_pct=Decimal("30"))])
        cache = simulator.run(profile, [EnableResponseCache(hit_rate=Decimal("0.5"))])
        both = simulator.run(
            profile,
            [CompressPrompt(reduction_pct=Decimal("30")),
             EnableResponseCache(hit_rate=Decimal("0.5"))],
        )
        assert both.monthly_savings < compress.monthly_savings + cache.monthly_savings
        assert both.savings_pct < Decimal("100")

    def test_prompt_cache_moves_static_tokens_to_the_cheaper_class(
        self, profile: WorkloadProfile
    ) -> None:
        result = Simulator().run(profile, [EnablePromptCache(hit_rate=Decimal("1.0"))])
        assert result.monthly_savings > Decimal("0")
        assert result.quality_delta == Decimal("0")

    def test_compression_does_not_touch_static_or_rag_regions(
        self, profile: WorkloadProfile
    ) -> None:
        """Otherwise it double-counts against caching and RAG pruning."""
        result = Simulator().run(profile, [CompressPrompt(reduction_pct=Decimal("100"))])
        # 4k static + 3k RAG must survive a 100% compression of the dynamic body.
        assert result.projected_tokens_per_request >= 7_000

    def test_quality_regression_vetoes_a_large_saving(
        self, profile: WorkloadProfile
    ) -> None:
        result = Simulator().run(
            profile,
            [SwitchModel(provider=Provider.GROQ, model="llama-4-8b",
                         quality_delta=Decimal("-0.20"))],
        )
        assert result.recommendation.startswith("reject")

    def test_warns_when_context_exceeds_the_target_window(self) -> None:
        """A scenario that is not physically executable must say so, not just
        report a cheaper price."""
        profile = WorkloadProfile(
            provider=Provider.OPENAI, model="gpt-4.1",
            monthly_requests=1000, avg_input_tokens=400_000, avg_output_tokens=2000,
        )
        result = Simulator().run(
            profile,
            [SwitchModel(provider=Provider.ANTHROPIC, model="claude-haiku-4-5")],
        )
        assert any("window" in w for w in result.warnings)

    def test_warns_on_implausibly_large_savings(self, profile: WorkloadProfile) -> None:
        result = Simulator().run(
            profile, [EnableResponseCache(hit_rate=Decimal("0.95"))]
        )
        assert any("80%" in w for w in result.warnings)

    def test_batch_lever_flags_the_latency_cost(self, profile: WorkloadProfile) -> None:
        from app.services.simulator import BatchRequests

        result = Simulator().run(profile, [BatchRequests(eligible_ratio=Decimal("0.8"))])
        assert result.latency_multiplier > Decimal("2")
        assert any("latency" in w.lower() for w in result.warnings)

    def test_rag_reduction_reduces_context(self, profile: WorkloadProfile) -> None:
        result = Simulator().run(profile, [ReduceRagContext(reduction_pct=Decimal("50"))])
        assert result.projected_tokens_per_request < result.baseline_tokens_per_request


class TestRoi:
    def test_includes_engineering_time_in_payback(self) -> None:
        """A $200/mo saving costing 3 engineer-weeks is a net loss for two years."""
        result = roi(monthly_savings=200, implementation_hours=120, hourly_rate=120)
        assert result["payback_months"] > 12
        assert result["worth_doing"] is False

    def test_quick_config_change_pays_back_immediately(self) -> None:
        result = roi(monthly_savings=5000, implementation_hours=2, hourly_rate=120)
        assert result["payback_months"] < 1
        assert result["worth_doing"] is True

    def test_ongoing_cost_reduces_net_savings(self) -> None:
        result = roi(
            monthly_savings=1000, implementation_hours=10, ongoing_monthly_cost=900
        )
        assert result["net_monthly_savings"] == 100.0


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


class TestQualityGate:
    def _baseline(self, subject: str, score: float, n: int = 500) -> QualityBaseline:
        baseline = QualityBaseline(subject=subject)
        for dimension in (
            QualityDimension.GROUNDEDNESS,
            QualityDimension.TASK_SUCCESS,
            QualityDimension.ACCURACY,
            QualityDimension.HALLUCINATION_RATE,
        ):
            value = 1 - score if dimension is QualityDimension.HALLUCINATION_RATE else score
            baseline.measurements[dimension] = QualityMeasurement(
                dimension=dimension,
                score=Decimal(str(value)),
                sample_size=n,
                stddev=Decimal("0.05"),
            )
        return baseline

    def test_blocks_a_significant_regression_on_a_guarded_dimension(self) -> None:
        gate = QualityGate()
        verdict = gate.evaluate(self._baseline("v1", 0.92), self._baseline("v2", 0.70))
        assert verdict.blocked
        assert verdict.regressions

    def test_allows_an_equivalent_candidate(self) -> None:
        gate = QualityGate()
        verdict = gate.evaluate(self._baseline("v1", 0.90), self._baseline("v2", 0.90))
        assert not verdict.blocked

    def test_blocks_when_a_guarded_dimension_was_never_measured(self) -> None:
        """A change cannot be certified against a dimension with no data."""
        gate = QualityGate()
        candidate = QualityBaseline(subject="v2")
        candidate.measurements[QualityDimension.RELEVANCE] = QualityMeasurement(
            dimension=QualityDimension.RELEVANCE, score=Decimal("0.9"), sample_size=500
        )
        verdict = gate.evaluate(self._baseline("v1", 0.9), candidate)
        assert verdict.blocked
        assert any("not measured" in r for r in verdict.reasons)

    def test_blocks_on_insufficient_samples(self) -> None:
        gate = QualityGate(min_sample_size=100)
        verdict = gate.evaluate(
            self._baseline("v1", 0.9, n=500), self._baseline("v2", 0.9, n=10)
        )
        assert verdict.blocked
        assert any("Insufficient samples" in r for r in verdict.reasons)

    def test_hallucination_rate_is_inverted_correctly(self) -> None:
        """Lower is better — getting the sign wrong silently inverts the gate."""
        gate = QualityGate(guarded={QualityDimension.HALLUCINATION_RATE})
        baseline = QualityBaseline(subject="v1")
        baseline.measurements[QualityDimension.HALLUCINATION_RATE] = QualityMeasurement(
            dimension=QualityDimension.HALLUCINATION_RATE,
            score=Decimal("0.02"), sample_size=500, stddev=Decimal("0.01"),
        )
        worse = QualityBaseline(subject="v2")
        worse.measurements[QualityDimension.HALLUCINATION_RATE] = QualityMeasurement(
            dimension=QualityDimension.HALLUCINATION_RATE,
            score=Decimal("0.15"), sample_size=500, stddev=Decimal("0.01"),
        )
        assert gate.evaluate(baseline, worse).blocked
        assert not gate.evaluate(worse, baseline).blocked


class TestDeterministicScorers:
    def test_groundedness_rewards_context_supported_answers(self) -> None:
        context = ["The quarterly revenue increased by twelve percent to four million dollars"]
        grounded = "The quarterly revenue increased by twelve percent to four million dollars"
        invented = "The company was founded in nineteen eighty by two graduate students"
        assert groundedness_score(grounded, context) > groundedness_score(invented, context)

    def test_groundedness_is_zero_without_context(self) -> None:
        assert groundedness_score("anything at all here", []) == Decimal("0")

    def test_refusal_detection(self) -> None:
        assert refusal_detected("I cannot help with that request.")
        assert not refusal_detected("The answer is 42 because of the following reasons.")

    def test_build_baseline_computes_dispersion(self) -> None:
        baseline = build_baseline(
            "v1", {QualityDimension.ACCURACY: [Decimal("0.8"), Decimal("0.9"), Decimal("1.0")]}
        )
        measurement = baseline.measurements[QualityDimension.ACCURACY]
        assert measurement.sample_size == 3
        assert measurement.stddev > Decimal("0")
