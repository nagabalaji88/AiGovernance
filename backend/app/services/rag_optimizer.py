"""RAG cost optimization.

RAG is usually the largest single line item in an enterprise AI bill, because
every request drags `top_k * chunk_size` tokens of context to the model whether
or not those chunks are relevant. The levers are few and well understood; what
is missing in practice is the arithmetic connecting a parameter to a dollar.

## The central trade-off

Retrieval recall rises with `top_k`, but with sharply diminishing returns —
past roughly k=5 most added chunks are noise for typical enterprise corpora —
while cost rises strictly linearly. So the optimum is nearly always lower than
what teams ship, because k was tuned for recall alone with no cost term.

This module refuses to recommend a k reduction on cost grounds alone. It needs
either an observed relevance/citation signal showing the tail chunks are unused,
or it labels the recommendation as requiring an evaluation run first. Cutting k
blind is how a cost programme causes a quality incident and gets shut down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from app.domain.money import ZERO, quantize_cost, safe_div, to_decimal

#: Empirical recall-at-k curve for typical enterprise document corpora,
#: normalised so k=10 is 1.0. Used to estimate the quality cost of reducing k
#: when a tenant has no measured relevance data yet. Replaced per-tenant by
#: measured citation rates as soon as those exist.
RECALL_CURVE: dict[int, Decimal] = {
    1: Decimal("0.62"),
    2: Decimal("0.74"),
    3: Decimal("0.83"),
    4: Decimal("0.88"),
    5: Decimal("0.92"),
    6: Decimal("0.94"),
    7: Decimal("0.96"),
    8: Decimal("0.975"),
    10: Decimal("1.00"),
    15: Decimal("1.02"),
    20: Decimal("1.03"),
}


@dataclass(slots=True)
class RagConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 10
    reranker_enabled: bool = False
    hybrid_search: bool = False
    #: Compression/summarisation applied to retrieved chunks before injection.
    context_compression: bool = False
    embedding_model: str = "text-embedding-3-small"

    @property
    def effective_chunk_tokens(self) -> int:
        """Chunk plus its overlap — the real per-chunk token cost.

        Overlap is invisible in most RAG configs but is paid on every chunk of
        every request. At 25% overlap and k=10 it is 2.5 chunks of pure
        duplication per call.
        """
        return self.chunk_size + self.chunk_overlap

    @property
    def context_tokens(self) -> int:
        return self.effective_chunk_tokens * self.top_k

    @property
    def overlap_waste_tokens(self) -> int:
        return self.chunk_overlap * self.top_k


@dataclass(slots=True)
class RagOptimization:
    lever: str
    current: str
    proposed: str
    tokens_saved_per_request: int
    monthly_savings: Decimal
    estimated_recall_delta: Decimal
    confidence: Decimal
    requires_evaluation: bool
    rationale: str

    @property
    def quality_risk(self) -> str:
        if self.estimated_recall_delta >= ZERO:
            return "none"
        if self.estimated_recall_delta > Decimal("-0.03"):
            return "low"
        if self.estimated_recall_delta > Decimal("-0.08"):
            return "medium"
        return "high"

    def as_dict(self) -> dict[str, object]:
        return {
            "lever": self.lever,
            "current": self.current,
            "proposed": self.proposed,
            "tokens_saved_per_request": self.tokens_saved_per_request,
            "monthly_savings": float(self.monthly_savings),
            "estimated_recall_delta": float(self.estimated_recall_delta),
            "quality_risk": self.quality_risk,
            "confidence": float(self.confidence),
            "requires_evaluation": self.requires_evaluation,
            "rationale": self.rationale,
        }


def _recall_at(k: int) -> Decimal:
    """Interpolate the recall curve for an arbitrary k."""
    if k in RECALL_CURVE:
        return RECALL_CURVE[k]
    keys = sorted(RECALL_CURVE)
    if k < keys[0]:
        return RECALL_CURVE[keys[0]] * safe_div(Decimal(k), Decimal(keys[0]))
    if k > keys[-1]:
        return RECALL_CURVE[keys[-1]]
    lower = max(x for x in keys if x < k)
    upper = min(x for x in keys if x > k)
    span = Decimal(upper - lower)
    weight = safe_div(Decimal(k - lower), span)
    return RECALL_CURVE[lower] + (RECALL_CURVE[upper] - RECALL_CURVE[lower]) * weight


def optimize_top_k(
    config: RagConfig,
    *,
    input_rate_per_token: Decimal,
    monthly_requests: int,
    observed_citation_rate: dict[int, Decimal] | None = None,
    max_recall_loss: Decimal = Decimal("0.03"),
) -> RagOptimization | None:
    """Find the smallest k whose estimated recall loss stays within tolerance.

    `observed_citation_rate` maps rank position to the share of answers that
    actually cited a chunk at that rank. When present it is authoritative and
    the recommendation is safe to apply directly; without it we fall back to
    the generic curve and flag `requires_evaluation`.
    """
    if config.top_k <= 3:
        return None

    if observed_citation_rate:
        # Trim ranks that are essentially never cited. 2% is the noise floor
        # below which a position is not earning its tokens.
        useful = [rank for rank, rate in sorted(observed_citation_rate.items()) if rate >= Decimal("0.02")]
        proposed_k = max(3, max(useful) if useful else 3)
        requires_eval = False
        confidence = Decimal("0.85")
        recall_delta = -sum(
            (rate for rank, rate in observed_citation_rate.items() if rank > proposed_k),
            ZERO,
        )
        basis = f"Ranks beyond {proposed_k} were cited in under 2% of answers over the observed window."
    else:
        baseline = _recall_at(config.top_k)
        proposed_k = config.top_k
        for k in range(3, config.top_k):
            if baseline - _recall_at(k) <= max_recall_loss:
                proposed_k = k
                break
        requires_eval = True
        confidence = Decimal("0.5")
        recall_delta = _recall_at(proposed_k) - baseline
        basis = "Estimated from a generic recall-at-k curve; no per-tenant citation data yet."

    if proposed_k >= config.top_k:
        return None

    saved = config.effective_chunk_tokens * (config.top_k - proposed_k)
    monthly = quantize_cost(Decimal(saved) * to_decimal(input_rate_per_token) * Decimal(monthly_requests))
    return RagOptimization(
        lever="top_k",
        current=str(config.top_k),
        proposed=str(proposed_k),
        tokens_saved_per_request=saved,
        monthly_savings=monthly,
        estimated_recall_delta=recall_delta,
        confidence=confidence,
        requires_evaluation=requires_eval,
        rationale=(
            f"Reducing top_k from {config.top_k} to {proposed_k} removes {saved:,} tokens "
            f"from every request. {basis}"
        ),
    )


def optimize_overlap(
    config: RagConfig,
    *,
    input_rate_per_token: Decimal,
    monthly_requests: int,
) -> RagOptimization | None:
    """Trim excessive chunk overlap.

    Overlap exists to stop a semantic unit being split across a boundary. Past
    ~15% of chunk size it buys almost nothing and is paid on every chunk of
    every request — the highest-leverage, lowest-risk RAG saving available, and
    the one teams most consistently miss because overlap is set once at
    ingestion and never revisited.
    """
    target_overlap = int(config.chunk_size * 0.15)
    if config.chunk_overlap <= target_overlap:
        return None

    saved = (config.chunk_overlap - target_overlap) * config.top_k
    monthly = quantize_cost(Decimal(saved) * to_decimal(input_rate_per_token) * Decimal(monthly_requests))
    return RagOptimization(
        lever="chunk_overlap",
        current=f"{config.chunk_overlap} tokens ({config.chunk_overlap * 100 // config.chunk_size}%)",
        proposed=f"{target_overlap} tokens (15%)",
        tokens_saved_per_request=saved,
        monthly_savings=monthly,
        estimated_recall_delta=Decimal("-0.005"),
        confidence=Decimal("0.8"),
        requires_evaluation=False,
        rationale=(
            f"Overlap of {config.chunk_overlap} tokens on a {config.chunk_size}-token chunk is "
            f"duplicated across all {config.top_k} retrieved chunks. Reducing to 15% preserves "
            "boundary context at a fraction of the cost. Requires a re-index."
        ),
    )


def optimize_chunk_size(
    config: RagConfig,
    *,
    input_rate_per_token: Decimal,
    monthly_requests: int,
    avg_answer_span_tokens: int = 180,
) -> RagOptimization | None:
    """Right-size chunks against the length of text answers actually draw on.

    If answers typically cite a 180-token span but chunks are 1024 tokens, then
    roughly 82% of every retrieved chunk is padding. Smaller chunks with a
    reranker usually beat large chunks on both cost and precision; the caveat
    is that too-small chunks fragment tables and code, so we floor at 256.
    """
    ideal = max(256, int(avg_answer_span_tokens * 1.8))
    if config.chunk_size <= ideal * 1.3:
        return None

    proposed = ideal
    # Smaller chunks mean more of them are needed for the same coverage; assume
    # k rises by ~30% to hold recall, and net the two effects.
    new_k = min(config.top_k, max(3, int(config.top_k * 1.3)))
    current_tokens = config.context_tokens
    new_tokens = (proposed + int(proposed * 0.15)) * new_k
    saved = current_tokens - new_tokens
    if saved <= 0:
        return None

    monthly = quantize_cost(Decimal(saved) * to_decimal(input_rate_per_token) * Decimal(monthly_requests))
    return RagOptimization(
        lever="chunk_size",
        current=f"{config.chunk_size} tokens",
        proposed=f"{proposed} tokens (top_k {config.top_k} -> {new_k})",
        tokens_saved_per_request=saved,
        monthly_savings=monthly,
        estimated_recall_delta=Decimal("0.01"),
        confidence=Decimal("0.55"),
        requires_evaluation=True,
        rationale=(
            f"Answers typically draw on ~{avg_answer_span_tokens} tokens, but chunks are "
            f"{config.chunk_size}. Smaller chunks raise retrieval precision while cutting "
            "injected context. Requires a full re-index and an evaluation run; keep chunks "
            "above 256 tokens so tables and code blocks are not fragmented."
        ),
    )


def recommend_reranker(
    config: RagConfig,
    *,
    input_rate_per_token: Decimal,
    monthly_requests: int,
    reranker_cost_per_request: Decimal = Decimal("0.0005"),
) -> RagOptimization | None:
    """A reranker lets you over-retrieve cheaply then inject only the best few.

    Economics: retrieve k=20 from the vector store (free — it never touches the
    LLM), rerank, inject the top 4. You pay a small reranking fee and save the
    generation cost of 16 chunks. This is nearly always net positive at scale,
    and it *improves* precision, making it one of the rare optimizations with a
    positive quality delta.
    """
    if config.reranker_enabled or config.top_k <= 5:
        return None

    proposed_k = 4
    saved_tokens = config.effective_chunk_tokens * (config.top_k - proposed_k)
    gross = Decimal(saved_tokens) * to_decimal(input_rate_per_token)
    net_per_request = gross - reranker_cost_per_request
    if net_per_request <= ZERO:
        return None

    monthly = quantize_cost(net_per_request * Decimal(monthly_requests))
    return RagOptimization(
        lever="reranker",
        current=f"no reranker, top_k={config.top_k}",
        proposed=f"cross-encoder reranker, retrieve {config.top_k * 2}, inject {proposed_k}",
        tokens_saved_per_request=saved_tokens,
        monthly_savings=monthly,
        estimated_recall_delta=Decimal("0.02"),
        confidence=Decimal("0.7"),
        requires_evaluation=True,
        rationale=(
            "Over-retrieve from the vector store (which costs nothing per token), rerank, and "
            f"inject only the best {proposed_k} chunks. Net of the reranker fee this saves "
            f"{quantize_cost(net_per_request)} per request and typically improves precision, "
            "because the cross-encoder sees the query and chunk together rather than comparing "
            "independent embeddings."
        ),
    )


def recommend_embedding_downgrade(
    config: RagConfig,
    *,
    current_rate: Decimal,
    candidate_model: str,
    candidate_rate: Decimal,
    monthly_embedding_tokens: int,
    quality_delta: Decimal = Decimal("-0.02"),
) -> RagOptimization | None:
    """Swap to a cheaper embedding model where retrieval quality permits.

    Embedding cost is usually a small share of total spend, so this matters
    most for high-ingestion workloads (continuous document indexing) rather
    than query-heavy ones. Flagged as requiring evaluation always: changing
    embedding models invalidates the entire index and cannot be rolled back
    without a full re-embed, which makes it the most operationally expensive
    recommendation in the catalog.
    """
    if candidate_rate >= current_rate:
        return None
    saved = (to_decimal(current_rate) - to_decimal(candidate_rate)) * Decimal(monthly_embedding_tokens)
    monthly = quantize_cost(saved)
    if monthly < Decimal("10"):
        return None
    return RagOptimization(
        lever="embedding_model",
        current=config.embedding_model,
        proposed=candidate_model,
        tokens_saved_per_request=0,
        monthly_savings=monthly,
        estimated_recall_delta=quality_delta,
        confidence=Decimal("0.6"),
        requires_evaluation=True,
        rationale=(
            f"Switching from {config.embedding_model} to {candidate_model} saves {monthly} "
            "per month at current embedding volume. This requires re-embedding the entire "
            "corpus and cannot be partially rolled out — run a retrieval evaluation on a "
            "representative query set against a shadow index first."
        ),
    )


@dataclass(slots=True)
class RagAdvisory:
    config: RagConfig
    optimizations: list[RagOptimization] = field(default_factory=list)

    @property
    def total_monthly_savings(self) -> Decimal:
        """Sum of *non-conflicting* savings.

        top_k, chunk_size and reranker all act on the same token budget, so
        naively summing them triple-counts. We take the single largest of that
        mutually-exclusive group plus the independent levers (overlap,
        embedding model). Overstating savings is the fastest way to lose FinOps
        credibility, and it is a mistake that only surfaces after the fact.
        """
        exclusive = {"top_k", "chunk_size", "reranker"}
        best_exclusive = max(
            (o.monthly_savings for o in self.optimizations if o.lever in exclusive),
            default=ZERO,
        )
        independent = sum((o.monthly_savings for o in self.optimizations if o.lever not in exclusive), ZERO)
        return quantize_cost(best_exclusive + independent)

    @property
    def safe_savings(self) -> Decimal:
        """Savings available without an evaluation run — what a team can do today."""
        return quantize_cost(
            sum((o.monthly_savings for o in self.optimizations if not o.requires_evaluation), ZERO)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "current_context_tokens_per_request": self.config.context_tokens,
            "overlap_waste_tokens_per_request": self.config.overlap_waste_tokens,
            "total_monthly_savings": float(self.total_monthly_savings),
            "safe_savings_no_evaluation_needed": float(self.safe_savings),
            "optimizations": [o.as_dict() for o in self.optimizations],
        }


def advise(
    config: RagConfig,
    *,
    input_rate_per_token: Decimal | float | str,
    monthly_requests: int,
    observed_citation_rate: dict[int, Decimal] | None = None,
    avg_answer_span_tokens: int = 180,
) -> RagAdvisory:
    rate = to_decimal(input_rate_per_token)
    candidates = [
        optimize_overlap(config, input_rate_per_token=rate, monthly_requests=monthly_requests),
        optimize_top_k(
            config,
            input_rate_per_token=rate,
            monthly_requests=monthly_requests,
            observed_citation_rate=observed_citation_rate,
        ),
        optimize_chunk_size(
            config,
            input_rate_per_token=rate,
            monthly_requests=monthly_requests,
            avg_answer_span_tokens=avg_answer_span_tokens,
        ),
        recommend_reranker(config, input_rate_per_token=rate, monthly_requests=monthly_requests),
    ]
    optimizations = [c for c in candidates if c is not None]
    optimizations.sort(key=lambda o: o.monthly_savings, reverse=True)
    return RagAdvisory(config=config, optimizations=optimizations)
