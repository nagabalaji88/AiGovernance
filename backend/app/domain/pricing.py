"""Model rate cards and the pricing resolution logic.

## Why rate cards are versioned rather than mutated in place

Providers change prices, and they do so without warning. If we stored a single
mutable `price_per_token` column on the model row, then re-running last
quarter's chargeback report after a price change would silently restate history
— finance would see numbers that no longer match the invoices they already
paid. Every rate card therefore carries `[effective_from, effective_to)` and
cost resolution is always keyed on the *timestamp of the usage event*, never
"now". This makes historical recomputation idempotent, which is a hard
requirement for SOX-adjacent chargeback.

## Where the numbers come from

`SEED_RATE_CARDS` below is bootstrap data for local development and first boot.
In production the authoritative catalog lives in the `model_pricing` table and
is refreshed by the `sync_provider_pricing` Celery beat job, which pulls
published provider price lists and opens a new effective-dated row on change
(never an UPDATE). Rates are expressed **per million tokens** because that is
the unit every provider publishes in, which keeps the seed data auditable by
eye against a vendor pricing page.

These seed figures are approximate list prices and deliberately not treated as
ground truth: enterprise agreements, committed-use discounts and regional
Bedrock/Azure premiums all move them. `RateCard.discount_multiplier` carries
the negotiated adjustment per tenant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Final

from app.domain.enums import ModelType, Provider, TokenClass
from app.domain.money import ZERO, to_decimal

#: Provider price lists are quoted per 1M tokens; we store the same unit.
TOKENS_PER_PRICING_UNIT: Final[Decimal] = Decimal("1000000")

_EPOCH: Final[datetime] = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class RateCard:
    """Effective-dated price list for one model.

    `rates` maps a token class to a price per 1M tokens. A class absent from
    the mapping is billed at zero — which is the correct behaviour for, say,
    `REASONING` tokens on a non-reasoning model, or `CACHED_INPUT` on a
    provider with no prompt caching.
    """

    provider: Provider
    model: str
    model_type: ModelType
    rates: dict[TokenClass, Decimal]
    #: Flat per-call charge, used by image/speech endpoints that bill per
    #: request or per generated asset rather than per token.
    per_request: Decimal = ZERO
    currency: str = "USD"
    effective_from: datetime = _EPOCH
    effective_to: datetime | None = None
    #: Negotiated adjustment, e.g. Decimal("0.85") for a 15% enterprise
    #: discount. Applied multiplicatively to every token class.
    discount_multiplier: Decimal = Decimal("1")
    #: Context window in tokens. Used by the router to reject candidates that
    #: physically cannot hold the request, and by the context analyser to
    #: compute window utilisation.
    context_window: int = 128_000
    max_output_tokens: int = 16_384
    #: Observed p50 latency per 1k output tokens, refreshed from telemetry.
    #: Seeded with a rough prior so the router is usable on a cold install.
    latency_ms_per_1k_output: int = 1_000
    #: Normalised 0-1 quality prior from public benchmarks, overridden per
    #: tenant by measured task-success once enough evaluations accumulate.
    quality_index: Decimal = Decimal("0.75")
    supports_prompt_cache: bool = False
    supports_batch: bool = False
    supports_vision: bool = False
    supports_tools: bool = True

    def is_effective_at(self, moment: datetime) -> bool:
        if moment < self.effective_from:
            return False
        return self.effective_to is None or moment < self.effective_to

    def rate_for(self, token_class: TokenClass) -> Decimal:
        """Price per single token for `token_class`, discount applied."""
        per_million = self.rates.get(token_class, ZERO)
        if per_million == ZERO:
            return ZERO
        return (per_million * self.discount_multiplier) / TOKENS_PER_PRICING_UNIT

    def blended_rate(self, output_ratio: Decimal = Decimal("0.25")) -> Decimal:
        """Single per-token price assuming a given output:total token mix.

        Used for fast candidate ranking in the router and for order-of-
        magnitude simulation, where resolving each token class separately would
        cost more than the precision is worth. `output_ratio` defaults to 0.25,
        the observed platform-wide median for chat workloads.
        """
        inp = self.rate_for(TokenClass.INPUT)
        out = self.rate_for(TokenClass.OUTPUT)
        return inp * (Decimal("1") - output_ratio) + out * output_ratio


def _card(
    provider: Provider,
    model: str,
    model_type: ModelType,
    *,
    inp: str = "0",
    out: str = "0",
    cached_in: str | None = None,
    cache_write: str | None = None,
    reasoning: str | None = None,
    embedding: str | None = None,
    image: str | None = None,
    audio_in: str | None = None,
    audio_out: str | None = None,
    per_request: str = "0",
    context_window: int = 128_000,
    max_output_tokens: int = 16_384,
    latency: int = 1_000,
    quality: str = "0.75",
    prompt_cache: bool = False,
    batch: bool = False,
    vision: bool = False,
    tools: bool = True,
) -> RateCard:
    """Terse constructor so the seed catalog stays readable as a table."""
    rates: dict[TokenClass, Decimal] = {}

    def put(cls_: TokenClass, raw: str | None) -> None:
        if raw is not None and raw != "0":
            rates[cls_] = to_decimal(raw)

    put(TokenClass.INPUT, inp)
    put(TokenClass.OUTPUT, out)
    put(TokenClass.CACHED_INPUT, cached_in)
    put(TokenClass.CACHE_WRITE, cache_write)
    put(TokenClass.REASONING, reasoning)
    put(TokenClass.EMBEDDING, embedding)
    put(TokenClass.IMAGE, image)
    put(TokenClass.AUDIO_INPUT, audio_in)
    put(TokenClass.AUDIO_OUTPUT, audio_out)

    return RateCard(
        provider=provider,
        model=model,
        model_type=model_type,
        rates=rates,
        per_request=to_decimal(per_request),
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        latency_ms_per_1k_output=latency,
        quality_index=to_decimal(quality),
        supports_prompt_cache=prompt_cache,
        supports_batch=batch,
        supports_vision=vision,
        supports_tools=tools,
    )


# ---------------------------------------------------------------------------
# Seed catalog. Prices are USD per 1M tokens, approximate published list rates.
# Reasoning models bill their hidden reasoning tokens at the output rate, which
# is why `reasoning` mirrors `out` for those entries — under-modelling this is
# a classic source of 3-5x budget overruns on o-series style workloads.
# ---------------------------------------------------------------------------
SEED_RATE_CARDS: Final[tuple[RateCard, ...]] = (
    # ---- OpenAI ----
    _card(
        Provider.OPENAI,
        "gpt-5",
        ModelType.REASONING,
        inp="1.25",
        out="10.00",
        cached_in="0.125",
        reasoning="10.00",
        context_window=400_000,
        max_output_tokens=128_000,
        latency=1400,
        quality="0.97",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    _card(
        Provider.OPENAI,
        "gpt-5-mini",
        ModelType.REASONING,
        inp="0.25",
        out="2.00",
        cached_in="0.025",
        reasoning="2.00",
        context_window=400_000,
        max_output_tokens=128_000,
        latency=800,
        quality="0.90",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    _card(
        Provider.OPENAI,
        "gpt-4.1",
        ModelType.MULTIMODAL,
        inp="2.00",
        out="8.00",
        cached_in="0.50",
        context_window=1_047_576,
        max_output_tokens=32_768,
        latency=900,
        quality="0.92",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    _card(
        Provider.OPENAI,
        "gpt-4.1-mini",
        ModelType.MULTIMODAL,
        inp="0.40",
        out="1.60",
        cached_in="0.10",
        context_window=1_047_576,
        max_output_tokens=32_768,
        latency=550,
        quality="0.86",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    _card(
        Provider.OPENAI,
        "gpt-4.1-nano",
        ModelType.CHAT,
        inp="0.10",
        out="0.40",
        cached_in="0.025",
        context_window=1_047_576,
        max_output_tokens=32_768,
        latency=350,
        quality="0.78",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    _card(
        Provider.OPENAI,
        "text-embedding-3-small",
        ModelType.EMBEDDING,
        embedding="0.02",
        context_window=8_191,
        max_output_tokens=0,
        latency=120,
        quality="0.80",
        batch=True,
        tools=False,
    ),
    _card(
        Provider.OPENAI,
        "text-embedding-3-large",
        ModelType.EMBEDDING,
        embedding="0.13",
        context_window=8_191,
        max_output_tokens=0,
        latency=180,
        quality="0.88",
        batch=True,
        tools=False,
    ),
    # ---- Anthropic ----
    _card(
        Provider.ANTHROPIC,
        "claude-fable-5",
        ModelType.REASONING,
        inp="5.00",
        out="25.00",
        cached_in="0.50",
        cache_write="6.25",
        reasoning="25.00",
        context_window=500_000,
        max_output_tokens=64_000,
        latency=1300,
        quality="0.98",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    _card(
        Provider.ANTHROPIC,
        "claude-opus-5",
        ModelType.REASONING,
        inp="5.00",
        out="25.00",
        cached_in="0.50",
        cache_write="6.25",
        reasoning="25.00",
        context_window=500_000,
        max_output_tokens=64_000,
        latency=1250,
        quality="0.96",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    _card(
        Provider.ANTHROPIC,
        "claude-sonnet-5",
        ModelType.MULTIMODAL,
        inp="3.00",
        out="15.00",
        cached_in="0.30",
        cache_write="3.75",
        context_window=500_000,
        max_output_tokens=64_000,
        latency=780,
        quality="0.93",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    _card(
        Provider.ANTHROPIC,
        "claude-haiku-4-5",
        ModelType.CHAT,
        inp="1.00",
        out="5.00",
        cached_in="0.10",
        cache_write="1.25",
        context_window=200_000,
        max_output_tokens=64_000,
        latency=380,
        quality="0.84",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    # ---- Google ----
    _card(
        Provider.GOOGLE_GEMINI,
        "gemini-2.5-pro",
        ModelType.MULTIMODAL,
        inp="1.25",
        out="10.00",
        cached_in="0.31",
        context_window=1_048_576,
        max_output_tokens=65_536,
        latency=1100,
        quality="0.93",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    _card(
        Provider.GOOGLE_GEMINI,
        "gemini-2.5-flash",
        ModelType.MULTIMODAL,
        inp="0.30",
        out="2.50",
        cached_in="0.075",
        context_window=1_048_576,
        max_output_tokens=65_536,
        latency=420,
        quality="0.85",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    _card(
        Provider.GOOGLE_GEMINI,
        "gemini-2.5-flash-lite",
        ModelType.CHAT,
        inp="0.10",
        out="0.40",
        context_window=1_048_576,
        max_output_tokens=65_536,
        latency=260,
        quality="0.76",
        batch=True,
        vision=True,
    ),
    # ---- Azure OpenAI (same list price, different commercial envelope) ----
    _card(
        Provider.AZURE_OPENAI,
        "gpt-4.1",
        ModelType.MULTIMODAL,
        inp="2.00",
        out="8.00",
        cached_in="0.50",
        context_window=1_047_576,
        max_output_tokens=32_768,
        latency=950,
        quality="0.92",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    _card(
        Provider.AZURE_OPENAI,
        "gpt-4.1-mini",
        ModelType.MULTIMODAL,
        inp="0.40",
        out="1.60",
        cached_in="0.10",
        context_window=1_047_576,
        max_output_tokens=32_768,
        latency=600,
        quality="0.86",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    # ---- Bedrock ----
    _card(
        Provider.AWS_BEDROCK,
        "anthropic.claude-sonnet-5",
        ModelType.MULTIMODAL,
        inp="3.00",
        out="15.00",
        cached_in="0.30",
        context_window=500_000,
        max_output_tokens=64_000,
        latency=850,
        quality="0.93",
        prompt_cache=True,
        batch=True,
        vision=True,
    ),
    _card(
        Provider.AWS_BEDROCK,
        "meta.llama-4-70b",
        ModelType.CHAT,
        inp="0.72",
        out="0.72",
        context_window=128_000,
        max_output_tokens=8_192,
        latency=500,
        quality="0.79",
        batch=True,
    ),
    # ---- Cost-optimised / OSS hosts ----
    _card(
        Provider.MISTRAL,
        "mistral-large-2",
        ModelType.CHAT,
        inp="2.00",
        out="6.00",
        context_window=128_000,
        max_output_tokens=8_192,
        latency=700,
        quality="0.85",
    ),
    _card(
        Provider.MISTRAL,
        "mistral-small-3",
        ModelType.CHAT,
        inp="0.20",
        out="0.60",
        context_window=128_000,
        max_output_tokens=8_192,
        latency=320,
        quality="0.75",
    ),
    _card(
        Provider.COHERE,
        "command-a",
        ModelType.CHAT,
        inp="2.50",
        out="10.00",
        context_window=256_000,
        max_output_tokens=8_192,
        latency=650,
        quality="0.84",
    ),
    _card(
        Provider.COHERE,
        "embed-v4",
        ModelType.EMBEDDING,
        embedding="0.12",
        context_window=128_000,
        max_output_tokens=0,
        latency=140,
        quality="0.87",
        tools=False,
    ),
    _card(
        Provider.DEEPSEEK,
        "deepseek-v3",
        ModelType.CHAT,
        inp="0.27",
        out="1.10",
        cached_in="0.07",
        context_window=128_000,
        max_output_tokens=8_192,
        latency=600,
        quality="0.83",
        prompt_cache=True,
    ),
    _card(
        Provider.DEEPSEEK,
        "deepseek-r1",
        ModelType.REASONING,
        inp="0.55",
        out="2.19",
        reasoning="2.19",
        context_window=128_000,
        max_output_tokens=32_768,
        latency=1600,
        quality="0.88",
    ),
    _card(
        Provider.GROQ,
        "llama-4-70b",
        ModelType.CHAT,
        inp="0.59",
        out="0.79",
        context_window=128_000,
        max_output_tokens=8_192,
        latency=90,
        quality="0.79",
    ),
    _card(
        Provider.GROQ,
        "llama-4-8b",
        ModelType.CHAT,
        inp="0.05",
        out="0.08",
        context_window=128_000,
        max_output_tokens=8_192,
        latency=45,
        quality="0.66",
    ),
    _card(
        Provider.TOGETHER,
        "qwen-3-72b",
        ModelType.CHAT,
        inp="0.90",
        out="0.90",
        context_window=128_000,
        max_output_tokens=8_192,
        latency=420,
        quality="0.81",
    ),
    _card(
        Provider.FIREWORKS,
        "llama-4-405b",
        ModelType.CHAT,
        inp="3.00",
        out="3.00",
        context_window=128_000,
        max_output_tokens=8_192,
        latency=900,
        quality="0.87",
    ),
    # ---- Self-hosted. Token rates are zero by construction: the cost is
    # amortised GPU time, computed by InfrastructureCostModel, not metered
    # tokens. Recording them at zero here keeps the resolution path uniform. ----
    _card(
        Provider.VLLM,
        "llama-4-70b-local",
        ModelType.LOCAL,
        context_window=128_000,
        max_output_tokens=8_192,
        latency=300,
        quality="0.79",
    ),
    _card(
        Provider.OLLAMA,
        "llama-4-8b-local",
        ModelType.LOCAL,
        context_window=128_000,
        max_output_tokens=8_192,
        latency=250,
        quality="0.66",
    ),
)


class PricingCatalog:
    """In-memory, time-aware index over rate cards.

    Held as a process-local singleton refreshed from Postgres on an interval.
    Cost resolution is on the hot ingestion path (target: 50k events/sec/pod),
    so a per-event database round trip to fetch a price is not viable — this
    index makes resolution an O(k) scan over the handful of cards for one
    model, with k = number of historical price changes, typically < 10.
    """

    def __init__(self, cards: tuple[RateCard, ...] | list[RateCard] = SEED_RATE_CARDS) -> None:
        self._by_key: dict[tuple[Provider, str], list[RateCard]] = {}
        self._all: list[RateCard] = []
        for card in cards:
            self.add(card)

    def add(self, card: RateCard) -> None:
        self._all.append(card)
        self._by_key.setdefault((card.provider, card.model), []).append(card)
        # Newest first: the common query is "price as of now", so the first
        # effective match is usually the first element.
        self._by_key[(card.provider, card.model)].sort(key=lambda c: c.effective_from, reverse=True)

    def resolve(self, provider: Provider, model: str, at: datetime | None = None) -> RateCard | None:
        """Rate card in force for `model` at `at` (default: now, UTC)."""
        moment = at or datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        for card in self._by_key.get((provider, model), ()):
            if card.is_effective_at(moment):
                return card
        return None

    def all_cards(self, at: datetime | None = None) -> list[RateCard]:
        moment = at or datetime.now(UTC)
        return [c for c in self._all if c.is_effective_at(moment)]

    def candidates(
        self,
        *,
        model_types: set[ModelType] | None = None,
        providers: set[Provider] | None = None,
        min_context: int = 0,
        requires_vision: bool = False,
        requires_tools: bool = False,
        at: datetime | None = None,
    ) -> list[RateCard]:
        """Cards satisfying hard constraints. Used as the router's input set."""
        out: list[RateCard] = []
        for card in self.all_cards(at):
            if model_types and card.model_type not in model_types:
                continue
            if providers and card.provider not in providers:
                continue
            if card.context_window < min_context:
                continue
            if requires_vision and not card.supports_vision:
                continue
            if requires_tools and not card.supports_tools:
                continue
            out.append(card)
        return out


#: Process-wide default catalog. Replaced at startup by the DB-backed refresh.
default_catalog: Final[PricingCatalog] = PricingCatalog()


@dataclass(frozen=True, slots=True)
class InfrastructureCostModel:
    """Cost model for self-hosted inference.

    A GPU-hour is a sunk, continuously-burning cost; tokens are the only
    proxy we have for apportioning it to a consumer. We therefore convert
    occupied GPU-seconds into dollars and attribute that to the request that
    occupied them, adding energy and network on top. Idle capacity is
    deliberately *not* attributed to any request — it surfaces separately as
    the `gpu_idle_waste` metric, because hiding idle cost inside per-request
    rates makes self-hosting look falsely expensive and blocks the (often
    correct) decision to consolidate onto fewer, busier nodes.
    """

    gpu_hourly_rate: Decimal = Decimal("3.90")  # e.g. A100-80GB on-demand
    gpu_count: int = 1
    energy_kwh_rate: Decimal = Decimal("0.14")
    gpu_kw_draw: Decimal = Decimal("0.40")
    #: Datacentre overhead multiplier (cooling, networking, facility).
    pue: Decimal = Decimal("1.4")
    network_gb_rate: Decimal = Decimal("0.09")
    storage_gb_month_rate: Decimal = Decimal("0.10")

    def compute_cost(
        self, *, gpu_seconds: Decimal, network_gb: Decimal = ZERO
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Return `(compute, energy, network)` cost for one request."""
        hours = gpu_seconds / Decimal("3600")
        compute = hours * self.gpu_hourly_rate * Decimal(self.gpu_count)
        energy = hours * self.gpu_kw_draw * Decimal(self.gpu_count) * self.pue * self.energy_kwh_rate
        network = network_gb * self.network_gb_rate
        return compute, energy, network


@dataclass(slots=True)
class PricingChange:
    """Audit record emitted when a rate card supersedes another."""

    provider: Provider
    model: str
    token_class: TokenClass
    old_rate: Decimal
    new_rate: Decimal
    effective_from: datetime
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def pct_delta(self) -> Decimal:
        if self.old_rate == ZERO:
            return ZERO
        return ((self.new_rate - self.old_rate) / self.old_rate) * Decimal("100")
