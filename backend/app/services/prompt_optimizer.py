"""Prompt optimization engine.

Finds token waste inside prompt text and quantifies what removing it is worth.

## Privacy posture

This module accepts raw prompt text, but the platform is designed so that it
*never has to store it*. Analysis runs either (a) in the tenant's own process
via the SDK, which ships back only the resulting metrics and fingerprints, or
(b) server-side against text the tenant explicitly submitted to the prompt
studio. The `PromptAnalysis` result carries no prompt content — only counts,
offsets and hashes — so it is safe to persist in the shared analytics store.
This is what keeps the platform out of the customer's PII/data-residency scope
while still being able to prove savings.

## Detection approach

Deterministic text analysis, not an LLM. Considered using a model to critique
prompts, and rejected it as the primary mechanism: a cost-optimization product
whose analysis path itself burns tokens at scale is self-defeating, results are
non-reproducible across runs, and a finding of "this is verbose" is not
auditable. Deterministic detectors are instant, free, reproducible, and each
finding points at exact character offsets a human can verify. An optional LLM
pass is offered for *rewriting* a prompt once a human has chosen to act, which
is a bounded, opt-in, per-prompt cost rather than a continuous one.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

from app.domain.money import ZERO, quantize_cost, safe_div, to_decimal

#: Empirical tokens-per-character for English prose across common BPE
#: tokenizers. Used only when an exact tokenizer is unavailable; the SDK path
#: always supplies exact counts from the provider's own tokenizer, and this
#: estimate is flagged as approximate wherever it is used.
CHARS_PER_TOKEN = Decimal("3.8")

#: Filler that reliably adds tokens without changing model behaviour. Kept
#: conservative — phrases that *do* affect output ("think step by step",
#: "be concise") are deliberately excluded, since stripping them is a quality
#: regression disguised as a saving.
FILLER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bplease\s+(?:kindly\s+)?", "polite filler"),
    (r"\bi\s+would\s+like\s+you\s+to\b", "verbose instruction preamble"),
    (r"\bit\s+is\s+important\s+(?:to\s+note\s+)?that\b", "emphasis filler"),
    (r"\bas\s+(?:an|a)\s+AI\s+(?:language\s+)?model\b", "redundant role framing"),
    (r"\bmake\s+sure\s+(?:that\s+)?you\b", "verbose imperative"),
    (r"\bin\s+order\s+to\b", "verbose connective"),
    (r"\bdue\s+to\s+the\s+fact\s+that\b", "verbose connective"),
    (r"\bat\s+this\s+point\s+in\s+time\b", "verbose temporal"),
    (r"\bfor\s+the\s+purpose\s+of\b", "verbose connective"),
    (r"\byou\s+are\s+required\s+to\b", "verbose imperative"),
    (r"\b(?:very|really|quite|extremely|highly)\s+", "intensifier"),
)

#: Instructions restated in different words. Detected as near-duplicate
#: sentences rather than exact matches, since restatement is usually
#: paraphrase.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9']+")


class Finding:
    """Base for a single optimization opportunity."""

    __slots__ = ("confidence", "detail", "kind", "severity", "span", "tokens_saved")

    def __init__(
        self,
        kind: str,
        detail: str,
        tokens_saved: int,
        *,
        confidence: Decimal = Decimal("0.8"),
        span: tuple[int, int] | None = None,
        severity: str = "medium",
    ) -> None:
        self.kind = kind
        self.detail = detail
        self.tokens_saved = tokens_saved
        self.confidence = confidence
        self.span = span
        self.severity = severity

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "tokens_saved": self.tokens_saved,
            "confidence": float(self.confidence),
            "span": list(self.span) if self.span else None,
            "severity": self.severity,
        }


@dataclass(slots=True)
class PromptAnalysis:
    """Content-free result of analysing one prompt."""

    fingerprint: str
    total_tokens: int
    findings: list[Finding] = field(default_factory=list)
    #: Sum of savings, de-overlapped so overlapping findings are not
    #: double-counted — the most common way these tools overstate their value.
    recoverable_tokens: int = 0
    static_tokens: int = 0
    dynamic_tokens: int = 0
    readability_penalty: Decimal = ZERO

    @property
    def compression_ratio(self) -> Decimal:
        """Share of the prompt that could be removed, 0-100."""
        return safe_div(Decimal(self.recoverable_tokens), Decimal(self.total_tokens)) * Decimal("100")

    @property
    def cacheable_ratio(self) -> Decimal:
        return safe_div(Decimal(self.static_tokens), Decimal(self.total_tokens)) * Decimal("100")

    def savings(self, rate_per_token: Decimal, calls_per_month: int) -> Decimal:
        """Monthly dollar value of acting on every finding."""
        return quantize_cost(Decimal(self.recoverable_tokens) * rate_per_token * Decimal(calls_per_month))

    def as_dict(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "total_tokens": self.total_tokens,
            "recoverable_tokens": self.recoverable_tokens,
            "compression_ratio": float(self.compression_ratio),
            "cacheable_ratio": float(self.cacheable_ratio),
            "static_tokens": self.static_tokens,
            "dynamic_tokens": self.dynamic_tokens,
            "findings": [f.as_dict() for f in self.findings],
        }


def fingerprint(text: str) -> str:
    """Stable, content-free identifier for a prompt.

    Normalises whitespace and case first so that cosmetic edits do not fork the
    identity of a prompt across versions. SHA-256 truncated to 32 hex chars:
    collision probability is negligible at any plausible prompt cardinality,
    and the shorter key halves index size on a very hot column.
    """
    normalised = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]


def estimate_tokens(text: str) -> int:
    """Approximate token count. Exact counts come from the SDK where available."""
    if not text:
        return 0
    return max(1, int(Decimal(len(text)) / CHARS_PER_TOKEN))


def _tokens_for(fragment: str) -> int:
    return max(1, estimate_tokens(fragment))


def _sentences(text: str) -> list[tuple[str, int, int]]:
    """Sentences with their character offsets, so findings can point at spans."""
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for raw in _SENTENCE_SPLIT.split(text):
        if not raw.strip():
            cursor += len(raw) + 1
            continue
        start = text.find(raw, cursor)
        if start < 0:
            start = cursor
        out.append((raw.strip(), start, start + len(raw)))
        cursor = start + len(raw)
    return out


def _jaccard(a: set[str], b: set[str]) -> Decimal:
    if not a or not b:
        return ZERO
    intersection = len(a & b)
    union = len(a | b)
    return safe_div(Decimal(intersection), Decimal(union))


def find_filler(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for pattern, label in FILLER_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            fragment = match.group(0)
            findings.append(
                Finding(
                    kind="filler_phrase",
                    # The label describes the *category* of filler, never the
                    # matched text. Findings are persisted in the shared
                    # analytics store, so they must stay content-free — see the
                    # privacy note in the module docstring.
                    detail=f"Removable {label} at characters {match.start()}-{match.end()}.",
                    tokens_saved=_tokens_for(fragment),
                    confidence=Decimal("0.9"),
                    span=(match.start(), match.end()),
                    severity="low",
                )
            )
    return findings


def find_repeated_instructions(text: str, *, similarity: Decimal = Decimal("0.6")) -> list[Finding]:
    """Sentences that restate an earlier instruction.

    Jaccard over word sets rather than embeddings: it needs no model, runs in
    microseconds, and for the specific case of *restated instructions* (which
    reuse most of the same vocabulary) it is nearly as effective. Semantic
    paraphrase with disjoint vocabulary is missed — an accepted limitation,
    documented rather than hidden, since the alternative is an embedding call
    per sentence on every analysis.
    """
    sentences = _sentences(text)
    findings: list[Finding] = []
    seen: list[tuple[set[str], int]] = []
    for sentence, start, end in sentences:
        words = set(_WORD.findall(sentence.lower()))
        if len(words) < 4:
            continue
        for prior_words, prior_offset in seen:
            overlap = _jaccard(words, prior_words)
            if overlap >= similarity:
                findings.append(
                    Finding(
                        kind="repeated_instruction",
                        # Offsets, not text: the UI highlights both spans in the
                        # user's own editor, so the reviewer sees the content
                        # without it ever leaving their browser.
                        detail=(
                            f"Restates the instruction at character {prior_offset} "
                            f"({overlap * 100:.0f}% word overlap); the model gains nothing "
                            "from the repetition."
                        ),
                        tokens_saved=_tokens_for(sentence),
                        confidence=Decimal("0.7"),
                        span=(start, end),
                        severity="medium",
                    )
                )
                break
        else:
            seen.append((words, start))
    return findings


def find_duplicate_context(text: str, *, min_block_chars: int = 120) -> list[Finding]:
    """Verbatim blocks appearing more than once.

    Almost always a RAG bug — the same chunk retrieved by two queries, or a
    document injected once as context and again as an example. High confidence
    and usually a large saving, so this is the first finding surfaced.
    """
    blocks = [b for b in re.split(r"\n\s*\n", text) if len(b.strip()) >= min_block_chars]
    counts = Counter(b.strip() for b in blocks)
    findings: list[Finding] = []
    for block, count in counts.items():
        if count < 2:
            continue
        start = text.find(block)
        findings.append(
            Finding(
                kind="duplicate_context",
                detail=(
                    f"A {len(block)}-character block appears {count} times verbatim. "
                    "Usually duplicate RAG retrieval or a double-injected document."
                ),
                tokens_saved=_tokens_for(block) * (count - 1),
                confidence=Decimal("0.95"),
                span=(start, start + len(block)) if start >= 0 else None,
                severity="high",
            )
        )
    return findings


def find_excess_few_shot(text: str, *, example_marker: str = "Example", max_useful: int = 3) -> list[Finding]:
    """Few-shot examples beyond the point of diminishing returns.

    Published few-shot scaling results converge by roughly 3-5 examples for
    most classification and extraction tasks; beyond that, each example is
    close to pure cost. Flagged as a *candidate* with an explicit instruction
    to A/B test, never auto-applied — the cut-off is task-dependent and this is
    exactly the kind of change that can quietly cost accuracy.
    """
    pattern = rf"\b{re.escape(example_marker)}\s*\d*\s*[:\-]"
    occurrences = [m.start() for m in re.finditer(pattern, text, re.IGNORECASE)]
    if len(occurrences) <= max_useful:
        return []
    excess = len(occurrences) - max_useful
    start = occurrences[max_useful]
    span_text = text[start:]
    return [
        Finding(
            kind="excess_few_shot",
            detail=(
                f"{len(occurrences)} few-shot examples present; accuracy typically plateaus "
                f"by {max_useful}. Trimming {excess} is a candidate — validate with an A/B test "
                "before rolling out."
            ),
            tokens_saved=int(_tokens_for(span_text) * 0.8),
            confidence=Decimal("0.5"),
            span=(start, len(text)),
            severity="medium",
        )
    ]


def find_verbose_formatting(text: str) -> list[Finding]:
    """Whitespace and separator padding.

    Individually trivial; at 10^7 calls/month a 40-token saving is real money,
    and it is a zero-risk change, which makes it a useful first win for a team
    that has not yet built confidence in the platform's recommendations.
    """
    findings: list[Finding] = []
    excess_blank = re.findall(r"\n{3,}", text)
    if excess_blank:
        saved = sum(_tokens_for(b) - 1 for b in excess_blank)
        if saved > 0:
            findings.append(
                Finding(
                    kind="verbose_formatting",
                    detail=f"{len(excess_blank)} runs of 3+ blank lines add tokens with no semantic effect.",
                    tokens_saved=saved,
                    confidence=Decimal("0.95"),
                    severity="low",
                )
            )
    separators = re.findall(r"[-=_*#]{20,}", text)
    if separators:
        findings.append(
            Finding(
                kind="verbose_formatting",
                detail=f"{len(separators)} long separator rules; a short heading conveys the same structure.",
                tokens_saved=sum(_tokens_for(s) for s in separators),
                confidence=Decimal("0.9"),
                severity="low",
            )
        )
    trailing = re.findall(r"[ \t]+\n", text)
    if len(trailing) > 20:
        findings.append(
            Finding(
                kind="verbose_formatting",
                detail=f"{len(trailing)} lines carry trailing whitespace.",
                tokens_saved=len(trailing) // 4,
                confidence=Decimal("0.99"),
                severity="low",
            )
        )
    return findings


def _deduplicate_savings(findings: list[Finding], total_tokens: int) -> int:
    """Sum savings without double-counting overlapping spans.

    Two detectors frequently fire on the same text (a duplicated block that is
    also verbose). Naively adding their savings can claim more tokens than the
    prompt contains, which destroys credibility the first time a customer
    checks the arithmetic. Span-covering resolves the overlap; findings with no
    span are added separately and the total is capped at 70% of the prompt.
    """
    spans = sorted((f.span for f in findings if f.span), key=lambda s: s[0])
    merged: list[tuple[int, int]] = []
    for span in spans:
        if merged and span[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
        else:
            merged.append(span)
    span_chars = sum(end - start for start, end in merged)
    span_tokens = int(Decimal(span_chars) / CHARS_PER_TOKEN)
    spanless = sum(f.tokens_saved for f in findings if not f.span)
    # A prompt cannot be 100% removable; the cap is a sanity bound against
    # pathological inputs, not a tuned parameter.
    return min(span_tokens + spanless, int(total_tokens * 0.7))


def analyse_prompt(
    text: str,
    *,
    exact_token_count: int | None = None,
    static_prefix_chars: int = 0,
) -> PromptAnalysis:
    """Run every detector over one prompt.

    `static_prefix_chars` marks the portion known to be call-invariant (system
    preamble, tool definitions), which the caller knows from its template
    structure. It drives the prompt-cache recommendation.
    """
    total = exact_token_count if exact_token_count is not None else estimate_tokens(text)
    findings: list[Finding] = []
    findings.extend(find_duplicate_context(text))
    findings.extend(find_repeated_instructions(text))
    findings.extend(find_filler(text))
    findings.extend(find_excess_few_shot(text))
    findings.extend(find_verbose_formatting(text))
    findings.sort(key=lambda f: f.tokens_saved, reverse=True)

    static_tokens = estimate_tokens(text[:static_prefix_chars]) if static_prefix_chars else 0
    return PromptAnalysis(
        fingerprint=fingerprint(text),
        total_tokens=total,
        findings=findings,
        recoverable_tokens=_deduplicate_savings(findings, total),
        static_tokens=static_tokens,
        dynamic_tokens=max(0, total - static_tokens),
    )


@dataclass(slots=True)
class PromptComparison:
    """A/B comparison between two prompt versions."""

    baseline_version: str
    candidate_version: str
    baseline_tokens: int
    candidate_tokens: int
    baseline_quality: Decimal
    candidate_quality: Decimal
    baseline_cost: Decimal
    candidate_cost: Decimal
    sample_size: int

    @property
    def token_delta_pct(self) -> Decimal:
        return safe_div(
            Decimal(self.candidate_tokens - self.baseline_tokens), Decimal(self.baseline_tokens)
        ) * Decimal("100")

    @property
    def quality_delta(self) -> Decimal:
        return self.candidate_quality - self.baseline_quality

    @property
    def cost_delta_pct(self) -> Decimal:
        return safe_div(self.candidate_cost - self.baseline_cost, self.baseline_cost) * Decimal("100")

    @property
    def is_statistically_meaningful(self) -> bool:
        """Guard against shipping on noise.

        A 200-sample floor is a blunt instrument, but it is the right kind of
        blunt: the alternative — teams promoting a prompt on 15 samples because
        the dashboard showed green — is how "optimization" becomes a quality
        incident. The exact threshold is configurable per tenant.
        """
        return self.sample_size >= 200

    def verdict(self, *, quality_tolerance: Decimal = Decimal("-0.02")) -> str:
        """Recommendation, with quality as a hard gate ahead of cost.

        Cost improvement never overrides a quality regression beyond tolerance.
        The default tolerance of -0.02 (2 percentage points of a 0-1 quality
        index) accepts genuine measurement noise while blocking real
        degradation.
        """
        if not self.is_statistically_meaningful:
            return "inconclusive: insufficient samples"
        if self.quality_delta < quality_tolerance:
            return "reject: quality regression beyond tolerance"
        if self.cost_delta_pct < Decimal("-5"):
            return "promote: cheaper at equal or better quality"
        if self.cost_delta_pct > Decimal("5"):
            return "reject: more expensive without quality gain"
        return "neutral: no material difference"


def score_prompt(analysis: PromptAnalysis) -> Decimal:
    """0-100 efficiency score for leaderboards and template ranking.

    Deliberately simple and stated openly in the UI. A composite score with
    opaque weights invites teams to argue with the score instead of fixing the
    prompt; this one is trivially explainable — start at 100, lose points for
    recoverable waste, gain a little for cacheable structure.
    """
    waste_penalty = analysis.compression_ratio * Decimal("1.2")
    cache_bonus = analysis.cacheable_ratio * Decimal("0.15")
    score = Decimal("100") - waste_penalty + cache_bonus
    return max(ZERO, min(Decimal("100"), score))


def suggest_compression(
    analysis: PromptAnalysis,
    *,
    rate_per_token: Decimal | float | str,
    monthly_calls: int,
) -> dict[str, object]:
    """Package an analysis into a costed, actionable recommendation payload."""
    rate = to_decimal(rate_per_token)
    monthly = analysis.savings(rate, monthly_calls)
    return {
        "fingerprint": analysis.fingerprint,
        "current_tokens": analysis.total_tokens,
        "optimised_tokens": analysis.total_tokens - analysis.recoverable_tokens,
        "reduction_pct": float(analysis.compression_ratio),
        "monthly_savings": float(monthly),
        "annual_savings": float(quantize_cost(monthly * Decimal("12"))),
        "efficiency_score": float(score_prompt(analysis)),
        "top_findings": [f.as_dict() for f in analysis.findings[:5]],
        "quality_risk": "low" if analysis.compression_ratio < Decimal("25") else "medium",
    }
