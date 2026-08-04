"""What-if simulation.

Lets a team price a change before making it: a different model, a compressed
prompt, caching turned on, RAG retuned, or several at once.

## Why simulations compose multiplicatively on tokens, not additively on cost

The naive implementation sums the individual savings of each change. That is
wrong and consistently overstates the result — if compression removes 30% of
the prompt and a cheaper model halves the rate, the combined saving is not
30% + 50% = 80%, it is 1 - (0.7 x 0.5) = 65%. The error compounds with each
additional lever and, applied to a four-lever scenario, can claim savings above
100%.

So the simulator models the *pipeline*: each lever transforms a token/rate
state, and cost is computed once at the end from the final state. This is
slightly more code than summing deltas and is the difference between a
simulation a CFO can act on and one that gets discredited the first time it is
checked against reality.

## Quality is simulated too, and gates the result

Every lever carries a quality impact estimate. The scenario's overall quality
delta is combined pessimistically (worst-case additive on the negatives),
because quality regressions from independent changes do stack, and a scenario
that saves 60% while degrading answers is a failure the platform should refuse
to recommend rather than present neutrally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Union

from app.domain.enums import Provider
from app.domain.money import ZERO, quantize_cost, safe_div, to_decimal
from app.domain.pricing import PricingCatalog, default_catalog
from app.domain.usage import TokenUsage


@dataclass
class WorkloadProfile:
    """The current state of a workload, as measured from real usage."""

    provider: Provider
    model: str
    monthly_requests: int
    avg_input_tokens: int
    avg_output_tokens: int
    avg_cached_input_tokens: int = 0
    avg_reasoning_tokens: int = 0
    #: Portion of input that is call-invariant (system prompt, tools,
    #: few-shot). Drives what prompt caching can actually cover.
    static_input_tokens: int = 0
    #: Portion of input that is retrieved RAG context.
    rag_input_tokens: int = 0
    avg_latency_ms: int = 1_000
    quality_index: Decimal = Decimal("0.90")
    #: Share of requests whose response could be served from a cache.
    cacheable_request_ratio: Decimal = ZERO

    def tokens(self) -> TokenUsage:
        return TokenUsage(
            input=self.avg_input_tokens,
            output=self.avg_output_tokens,
            cached_input=self.avg_cached_input_tokens,
            reasoning=self.avg_reasoning_tokens,
        )


@dataclass
class SimulationState:
    """Mutable token/rate state threaded through the lever pipeline."""

    input_tokens: Decimal
    output_tokens: Decimal
    cached_input_tokens: Decimal
    reasoning_tokens: Decimal
    static_tokens: Decimal
    rag_tokens: Decimal
    request_multiplier: Decimal = Decimal("1")
    provider: Provider = Provider.OPENAI
    model: str = ""
    quality_delta: Decimal = ZERO
    latency_multiplier: Decimal = Decimal("1")
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: WorkloadProfile) -> SimulationState:
        return cls(
            input_tokens=Decimal(profile.avg_input_tokens),
            output_tokens=Decimal(profile.avg_output_tokens),
            cached_input_tokens=Decimal(profile.avg_cached_input_tokens),
            reasoning_tokens=Decimal(profile.avg_reasoning_tokens),
            static_tokens=Decimal(profile.static_input_tokens),
            rag_tokens=Decimal(profile.rag_input_tokens),
            provider=profile.provider,
            model=profile.model,
        )

    def to_tokens(self) -> TokenUsage:
        return TokenUsage(
            input=max(0, int(self.input_tokens)),
            output=max(0, int(self.output_tokens)),
            cached_input=max(0, int(self.cached_input_tokens)),
            reasoning=max(0, int(self.reasoning_tokens)),
        )


class Lever:
    """A single simulated change. Subclasses transform the state in place."""

    name = "lever"

    def apply(self, state: SimulationState) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover - interface
        return self.name


@dataclass
class SwitchModel(Lever):
    """Route the workload to a different model."""

    provider: Provider
    model: str
    #: Measured or estimated quality difference vs. the current model.
    quality_delta: Decimal = ZERO
    name: str = "switch_model"

    def apply(self, state: SimulationState) -> None:
        state.provider = self.provider
        state.model = self.model
        state.quality_delta += self.quality_delta
        state.notes.append(f"Routed to {self.provider}/{self.model}.")

    def describe(self) -> str:
        return f"Switch to {self.provider}/{self.model}"


@dataclass
class CompressPrompt(Lever):
    """Remove a share of the non-static, non-RAG prompt tokens."""

    reduction_pct: Decimal
    quality_delta: Decimal = Decimal("-0.005")
    name: str = "compress_prompt"

    def apply(self, state: SimulationState) -> None:
        # Compression acts on the dynamic body only. Applying it to the whole
        # prompt double-counts against RAG pruning and prompt caching, which
        # target the other regions — the classic overlapping-savings error.
        compressible = max(ZERO, state.input_tokens - state.static_tokens - state.rag_tokens)
        removed = compressible * (self.reduction_pct / Decimal("100"))
        state.input_tokens -= removed
        state.quality_delta += self.quality_delta
        state.notes.append(
            f"Compressed the dynamic prompt body by {self.reduction_pct}% ({int(removed):,} tokens)."
        )

    def describe(self) -> str:
        return f"Compress prompt by {self.reduction_pct}%"


@dataclass
class EnablePromptCache(Lever):
    """Move the static prefix onto the provider's prompt cache."""

    hit_rate: Decimal = Decimal("0.85")
    name: str = "enable_prompt_cache"

    def apply(self, state: SimulationState) -> None:
        moved = state.static_tokens * self.hit_rate
        state.input_tokens -= moved
        state.cached_input_tokens += moved
        state.notes.append(
            f"Moved {int(moved):,} static tokens to the prompt cache at a "
            f"{self.hit_rate * 100:.0f}% hit rate."
        )

    def describe(self) -> str:
        return f"Enable prompt caching ({self.hit_rate * 100:.0f}% hit rate)"


@dataclass
class EnableResponseCache(Lever):
    """Serve a share of requests from cache, eliminating the call entirely."""

    hit_rate: Decimal
    name: str = "enable_response_cache"

    def apply(self, state: SimulationState) -> None:
        # This reduces the *number of billed requests*, not the size of each,
        # which is why it is a request multiplier rather than a token change.
        state.request_multiplier *= Decimal("1") - self.hit_rate
        state.notes.append(
            f"{self.hit_rate * 100:.0f}% of requests served from cache and never sent to the provider."
        )

    def describe(self) -> str:
        return f"Response cache at {self.hit_rate * 100:.0f}% hit rate"


@dataclass
class ReduceRagContext(Lever):
    """Cut retrieved context (lower top_k, trimmed overlap, reranking)."""

    reduction_pct: Decimal
    quality_delta: Decimal = Decimal("-0.01")
    name: str = "reduce_rag_context"

    def apply(self, state: SimulationState) -> None:
        removed = state.rag_tokens * (self.reduction_pct / Decimal("100"))
        state.rag_tokens -= removed
        state.input_tokens -= removed
        state.quality_delta += self.quality_delta
        state.notes.append(f"Reduced RAG context by {self.reduction_pct}% ({int(removed):,} tokens).")

    def describe(self) -> str:
        return f"Reduce RAG context by {self.reduction_pct}%"


@dataclass
class SummariseHistory(Lever):
    """Replace verbatim conversation history with a rolling summary."""

    reduction_pct: Decimal = Decimal("60")
    quality_delta: Decimal = Decimal("-0.015")
    name: str = "summarise_history"

    def apply(self, state: SimulationState) -> None:
        history = max(ZERO, state.input_tokens - state.static_tokens - state.rag_tokens)
        removed = history * (self.reduction_pct / Decimal("100"))
        state.input_tokens -= removed
        state.quality_delta += self.quality_delta
        state.notes.append(
            f"Rolling summarisation removed {int(removed):,} tokens of verbatim history. "
            "Note this adds a periodic summarisation call, priced separately."
        )

    def describe(self) -> str:
        return f"Summarise conversation history ({self.reduction_pct}% reduction)"


@dataclass
class BatchRequests(Lever):
    """Move eligible traffic to the provider's batch tier."""

    eligible_ratio: Decimal = Decimal("0.4")
    discount: Decimal = Decimal("0.5")
    name: str = "batch_requests"

    def apply(self, state: SimulationState) -> None:
        # Modelled as an effective request-count reduction, since batch tiers
        # discount the rate rather than the token count.
        effective = Decimal("1") - (self.eligible_ratio * self.discount)
        state.request_multiplier *= effective
        state.latency_multiplier *= Decimal("1") + self.eligible_ratio * Decimal("20")
        state.notes.append(
            f"{self.eligible_ratio * 100:.0f}% of traffic moved to the batch tier at a "
            f"{self.discount * 100:.0f}% discount. Batch latency is measured in hours — only "
            "applicable to workloads with no interactive user waiting."
        )

    def describe(self) -> str:
        return f"Batch {self.eligible_ratio * 100:.0f}% of requests"


@dataclass
class ScenarioResult:
    name: str
    baseline_monthly_cost: Decimal
    projected_monthly_cost: Decimal
    baseline_tokens_per_request: int
    projected_tokens_per_request: int
    quality_delta: Decimal
    latency_multiplier: Decimal
    levers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def monthly_savings(self) -> Decimal:
        return quantize_cost(self.baseline_monthly_cost - self.projected_monthly_cost)

    @property
    def annual_savings(self) -> Decimal:
        return quantize_cost(self.monthly_savings * Decimal("12"))

    @property
    def savings_pct(self) -> Decimal:
        return safe_div(self.monthly_savings, self.baseline_monthly_cost) * Decimal("100")

    @property
    def token_reduction_pct(self) -> Decimal:
        return safe_div(
            Decimal(self.baseline_tokens_per_request - self.projected_tokens_per_request),
            Decimal(max(self.baseline_tokens_per_request, 1)),
        ) * Decimal("100")

    @property
    def recommendation(self) -> str:
        """Verdict, with quality as a veto over cost.

        The thresholds are deliberately conservative. A 5-point quality drop is
        catastrophic for most enterprise use cases regardless of the saving,
        and presenting such a scenario as merely "a trade-off" invites someone
        to ship it.
        """
        if self.quality_delta < Decimal("-0.05"):
            return "reject: quality degradation outweighs any saving"
        if self.savings_pct < Decimal("5"):
            return "not worthwhile: saving does not justify the change"
        if self.quality_delta < Decimal("-0.02"):
            return "conditional: run an offline evaluation before rollout"
        return "recommended: material saving at acceptable quality"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "baseline_monthly_cost": float(self.baseline_monthly_cost),
            "projected_monthly_cost": float(self.projected_monthly_cost),
            "monthly_savings": float(self.monthly_savings),
            "annual_savings": float(self.annual_savings),
            "savings_pct": float(self.savings_pct),
            "token_reduction_pct": float(self.token_reduction_pct),
            "quality_delta": float(self.quality_delta),
            "latency_multiplier": float(self.latency_multiplier),
            "levers": self.levers,
            "notes": self.notes,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
        }


class Simulator:
    def __init__(self, catalog: PricingCatalog | None = None) -> None:
        self.catalog = catalog or default_catalog

    def run(self, profile: WorkloadProfile, levers: list[Lever], *, name: str = "scenario") -> ScenarioResult:
        """Apply levers in sequence and price the resulting state once."""
        baseline_cost = self._monthly_cost(
            profile.provider, profile.model, profile.tokens(), profile.monthly_requests
        )

        state = SimulationState.from_profile(profile)
        for lever in levers:
            lever.apply(state)

        projected_tokens = state.to_tokens()
        effective_requests = int(Decimal(profile.monthly_requests) * state.request_multiplier)
        projected_cost = self._monthly_cost(state.provider, state.model, projected_tokens, effective_requests)

        result = ScenarioResult(
            name=name,
            baseline_monthly_cost=baseline_cost,
            projected_monthly_cost=projected_cost,
            baseline_tokens_per_request=profile.tokens().total,
            projected_tokens_per_request=projected_tokens.total,
            quality_delta=state.quality_delta,
            latency_multiplier=state.latency_multiplier,
            levers=[lever.describe() for lever in levers],
            notes=state.notes,
        )
        result.warnings = self._warnings(profile, state, result)
        return result

    def compare(self, profile: WorkloadProfile, scenarios: dict[str, list[Lever]]) -> list[ScenarioResult]:
        """Run several scenarios and rank by saving, filtering rejected ones last."""
        results = [self.run(profile, levers, name=name) for name, levers in scenarios.items()]
        results.sort(
            key=lambda r: (not r.recommendation.startswith("reject"), r.monthly_savings),
            reverse=True,
        )
        return results

    def _monthly_cost(self, provider: Provider, model: str, tokens: TokenUsage, requests: int) -> Decimal:
        card = self.catalog.resolve(provider, model)
        if card is None:
            return ZERO
        per_request = ZERO
        for token_class, count in tokens.as_map().items():
            if count > 0:
                per_request += Decimal(count) * card.rate_for(token_class)
        per_request += card.per_request
        return quantize_cost(per_request * Decimal(requests))

    def _warnings(
        self, profile: WorkloadProfile, state: SimulationState, result: ScenarioResult
    ) -> list[str]:
        """Surface the assumptions most likely to make the simulation wrong."""
        warnings: list[str] = []
        card = self.catalog.resolve(state.provider, state.model)
        if card is None:
            warnings.append(f"No rate card for {state.provider}/{state.model}; projected cost is unreliable.")
            return warnings

        projected_context = int(state.input_tokens + state.output_tokens)
        if projected_context > card.context_window:
            warnings.append(
                f"Projected context of {projected_context:,} tokens exceeds "
                f"{state.model}'s {card.context_window:,}-token window. This scenario is not "
                "physically executable as configured."
            )
        if state.cached_input_tokens > 0 and not card.supports_prompt_cache:
            warnings.append(
                f"{state.model} has no prompt cache; the cached-token saving will not materialise."
            )
        if profile.static_input_tokens == 0 and state.cached_input_tokens > 0:
            warnings.append(
                "Prompt caching was simulated but the workload reports no static prefix. "
                "Instrument the static region before relying on this figure."
            )
        if result.latency_multiplier > Decimal("2"):
            warnings.append(
                f"Latency increases by {result.latency_multiplier:.1f}x. Confirm no interactive "
                "user is waiting on this workload."
            )
        if result.savings_pct > Decimal("80"):
            warnings.append(
                "Projected saving exceeds 80%. Verify the workload profile against measured "
                "usage — savings this large usually indicate a mis-specified baseline."
            )
        return warnings


def standard_scenarios(
    profile: WorkloadProfile, *, cheaper_model: Optional[tuple[Provider, str]] = None
) -> dict[str, list[Lever]]:
    """The default scenario set offered in the simulation UI.

    Chosen to span the risk spectrum: 'Safe wins' requires no evaluation and no
    quality risk, so a team can act on it immediately; 'Aggressive' shows the
    ceiling. Presenting both prevents the two failure modes — teams doing
    nothing because every option looks risky, and teams doing everything at
    once and causing a regression they cannot attribute.
    """
    scenarios: dict[str, list[Lever]] = {
        "Safe wins (no evaluation needed)": [
            EnablePromptCache(hit_rate=Decimal("0.85")),
            CompressPrompt(reduction_pct=Decimal("15"), quality_delta=ZERO),
        ],
        "Caching programme": [
            EnablePromptCache(hit_rate=Decimal("0.85")),
            EnableResponseCache(hit_rate=profile.cacheable_request_ratio or Decimal("0.2")),
        ],
        "Context discipline": [
            EnablePromptCache(hit_rate=Decimal("0.85")),
            ReduceRagContext(reduction_pct=Decimal("40")),
            SummariseHistory(),
        ],
    }
    if cheaper_model:
        provider, model = cheaper_model
        scenarios["Model downgrade"] = [
            SwitchModel(provider=provider, model=model, quality_delta=Decimal("-0.03"))
        ]
        scenarios["Aggressive (everything)"] = [
            SwitchModel(provider=provider, model=model, quality_delta=Decimal("-0.03")),
            EnablePromptCache(hit_rate=Decimal("0.85")),
            EnableResponseCache(hit_rate=profile.cacheable_request_ratio or Decimal("0.2")),
            ReduceRagContext(reduction_pct=Decimal("40")),
            CompressPrompt(reduction_pct=Decimal("25")),
        ]
    return scenarios


def roi(
    *,
    monthly_savings: Union[Decimal, float, str],
    implementation_hours: Union[Decimal, float, str],
    hourly_rate: Union[Decimal, float, str] = Decimal("120"),
    ongoing_monthly_cost: Union[Decimal, float, str] = ZERO,
) -> dict[str, float]:
    """Payback analysis for an optimization.

    Engineering time is included deliberately. A recommendation saving $200/mo
    that costs three engineer-weeks to implement is a net loss for two years,
    and a platform that reports only the saving will drive teams toward exactly
    those. Payback period is the number that should drive prioritisation.
    """
    savings = to_decimal(monthly_savings)
    net_monthly = savings - to_decimal(ongoing_monthly_cost)
    investment = to_decimal(implementation_hours) * to_decimal(hourly_rate)
    payback_months = safe_div(investment, net_monthly) if net_monthly > ZERO else Decimal("-1")
    first_year = net_monthly * Decimal("12") - investment
    return {
        "implementation_cost": float(quantize_cost(investment)),
        "net_monthly_savings": float(quantize_cost(net_monthly)),
        "payback_months": float(payback_months.quantize(Decimal("0.1"))) if payback_months > ZERO else None,  # type: ignore[dict-item]
        "first_year_net": float(quantize_cost(first_year)),
        "first_year_roi_pct": float(safe_div(first_year, investment) * Decimal("100")),
        "worth_doing": bool(net_monthly > ZERO and payback_months > ZERO and payback_months < Decimal("12")),
    }
