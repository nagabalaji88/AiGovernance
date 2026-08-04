"""Quality measurement — the guardrail that makes optimization safe.

Every other engine in this platform proposes ways to spend less. This one is
the counterweight: it measures whether a change degraded the output, and it has
veto power over recommendations.

## Why quality is a first-class, blocking concern

The failure mode of every AI cost programme is the same. Someone downgrades a
model, saves 60%, ships it, and three weeks later support tickets rise, a sales
demo goes badly, and the whole optimization effort is reverted and discredited
— including the changes that were genuinely free. The saving was real; the
platform simply had no way to see the cost.

So: no recommendation is marked `applied` without a quality baseline captured
before the change and a comparison after it. The `QualityGate` below is
consulted by the recommendation engine and returns a hard block, not a warning,
when a regression is detected on a metric the tenant marked as guarded.

## Measurement sources, in descending order of trust

1. **Business outcome** — task success, ticket deflection, conversion. Slow
   (days) and sparse, but it is ground truth and nothing else outranks it.
2. **Human review** — sampled expert grading. Expensive; reserved for
   high-stakes changes.
3. **Deterministic checks** — schema validity, citation presence, refusal
   detection, groundedness by string overlap against retrieved context. Free,
   instant, and surprisingly effective for RAG.
4. **LLM-as-judge** — flexible but itself costs tokens and carries known
   biases (verbosity preference, position bias, self-preference). Used as a
   *screen*, never as the sole gate for a production change, and always with
   the judge model held fixed across the comparison so the judge's own drift
   does not masquerade as a quality change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from statistics import NormalDist, fmean

from app.domain.enums import QualityDimension
from app.domain.money import ZERO, safe_div, to_decimal

#: Dimensions where a *lower* value is better, so delta signs must be flipped
#: before comparison. Getting this wrong silently inverts the gate.
INVERTED_DIMENSIONS: frozenset[QualityDimension] = frozenset(
    {QualityDimension.HALLUCINATION_RATE}
)

#: Default per-dimension regression tolerance, as an absolute delta on a 0-1
#: scale. Groundedness and hallucination are tighter than the rest because
#: those are the failures that produce incorrect answers rather than merely
#: worse ones.
DEFAULT_TOLERANCES: dict[QualityDimension, Decimal] = {
    QualityDimension.ACCURACY: Decimal("0.02"),
    QualityDimension.COMPLETENESS: Decimal("0.03"),
    QualityDimension.GROUNDEDNESS: Decimal("0.01"),
    QualityDimension.RELEVANCE: Decimal("0.03"),
    QualityDimension.CITATION_QUALITY: Decimal("0.03"),
    QualityDimension.HALLUCINATION_RATE: Decimal("0.01"),
    QualityDimension.USER_SATISFACTION: Decimal("0.05"),
    QualityDimension.TASK_SUCCESS: Decimal("0.02"),
}


@dataclass(slots=True)
class QualityMeasurement:
    """One dimension's score over a sample."""

    dimension: QualityDimension
    score: Decimal
    sample_size: int
    #: Standard deviation of the underlying sample, used for significance.
    stddev: Decimal = ZERO
    source: str = "deterministic"
    measured_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def standard_error(self) -> Decimal:
        if self.sample_size < 2:
            return ZERO
        return self.stddev / Decimal(str(self.sample_size**0.5))


@dataclass(slots=True)
class QualityBaseline:
    """Quality snapshot for a subject (model, prompt version, feature)."""

    subject: str
    measurements: dict[QualityDimension, QualityMeasurement] = field(default_factory=dict)
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def score(self, dimension: QualityDimension) -> Decimal | None:
        measurement = self.measurements.get(dimension)
        return measurement.score if measurement else None

    @property
    def composite(self) -> Decimal:
        """Single 0-1 quality index across all measured dimensions.

        Equal weighting, with inverted dimensions flipped first. Weighted
        composites are available per tenant, but the default is unweighted
        precisely because weights are where a quality metric gets quietly
        tuned until it stops blocking anything.
        """
        if not self.measurements:
            return ZERO
        scores = [
            (Decimal("1") - m.score) if m.dimension in INVERTED_DIMENSIONS else m.score
            for m in self.measurements.values()
        ]
        return to_decimal(fmean(float(s) for s in scores))


@dataclass(slots=True)
class DimensionComparison:
    dimension: QualityDimension
    baseline_score: Decimal
    candidate_score: Decimal
    tolerance: Decimal
    sample_size: int
    p_value: Decimal | None = None

    @property
    def raw_delta(self) -> Decimal:
        return self.candidate_score - self.baseline_score

    @property
    def effective_delta(self) -> Decimal:
        """Delta oriented so that negative always means worse."""
        if self.dimension in INVERTED_DIMENSIONS:
            return -self.raw_delta
        return self.raw_delta

    @property
    def is_regression(self) -> bool:
        return self.effective_delta < -self.tolerance

    @property
    def is_significant(self) -> bool:
        """Statistically distinguishable from no change.

        Without this, small samples produce a stream of phantom regressions and
        the gate becomes noise that teams learn to override — which is worse
        than having no gate, because it also trains them to override the real
        ones.
        """
        if self.p_value is None:
            return self.sample_size >= 100
        return self.p_value < Decimal("0.05")


@dataclass(slots=True)
class QualityVerdict:
    subject: str
    comparisons: list[DimensionComparison] = field(default_factory=list)
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)
    composite_delta: Decimal = ZERO

    @property
    def regressions(self) -> list[DimensionComparison]:
        return [c for c in self.comparisons if c.is_regression and c.is_significant]

    def as_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "blocked": self.blocked,
            "composite_delta": float(self.composite_delta),
            "reasons": self.reasons,
            "dimensions": [
                {
                    "dimension": str(c.dimension),
                    "baseline": float(c.baseline_score),
                    "candidate": float(c.candidate_score),
                    "delta": float(c.effective_delta),
                    "tolerance": float(c.tolerance),
                    "regression": c.is_regression,
                    "significant": c.is_significant,
                }
                for c in self.comparisons
            ],
        }


class QualityGate:
    """Blocks changes that degrade guarded quality dimensions."""

    def __init__(
        self,
        *,
        tolerances: dict[QualityDimension, Decimal] | None = None,
        guarded: set[QualityDimension] | None = None,
        min_sample_size: int = 100,
    ) -> None:
        self.tolerances = {**DEFAULT_TOLERANCES, **(tolerances or {})}
        # Guarding groundedness and task success by default: these are the two
        # that map most directly onto "the product still works".
        self.guarded = guarded or {
            QualityDimension.GROUNDEDNESS,
            QualityDimension.TASK_SUCCESS,
            QualityDimension.ACCURACY,
            QualityDimension.HALLUCINATION_RATE,
        }
        self.min_sample_size = min_sample_size

    def evaluate(
        self, baseline: QualityBaseline, candidate: QualityBaseline
    ) -> QualityVerdict:
        verdict = QualityVerdict(subject=candidate.subject)
        verdict.composite_delta = candidate.composite - baseline.composite

        for dimension, candidate_measurement in candidate.measurements.items():
            baseline_measurement = baseline.measurements.get(dimension)
            if baseline_measurement is None:
                continue
            comparison = DimensionComparison(
                dimension=dimension,
                baseline_score=baseline_measurement.score,
                candidate_score=candidate_measurement.score,
                tolerance=self.tolerances.get(dimension, Decimal("0.03")),
                sample_size=min(baseline_measurement.sample_size, candidate_measurement.sample_size),
                p_value=_welch_p_value(baseline_measurement, candidate_measurement),
            )
            verdict.comparisons.append(comparison)

        insufficient = [
            c for c in verdict.comparisons
            if c.dimension in self.guarded and c.sample_size < self.min_sample_size
        ]
        if insufficient:
            verdict.blocked = True
            verdict.reasons.append(
                "Insufficient samples on guarded dimensions ("
                + ", ".join(f"{c.dimension}: {c.sample_size}" for c in insufficient)
                + f"); {self.min_sample_size} required. Cannot certify no regression."
            )

        missing = self.guarded - set(candidate.measurements)
        if missing:
            verdict.blocked = True
            verdict.reasons.append(
                "Guarded dimensions were not measured: "
                + ", ".join(str(d) for d in sorted(missing, key=str))
                + ". A change cannot be certified against a dimension with no data."
            )

        for comparison in verdict.regressions:
            if comparison.dimension in self.guarded:
                verdict.blocked = True
                verdict.reasons.append(
                    f"{comparison.dimension} regressed by {abs(comparison.effective_delta):.3f}, "
                    f"beyond the {comparison.tolerance} tolerance (n={comparison.sample_size})."
                )
            else:
                verdict.reasons.append(
                    f"Non-blocking regression on {comparison.dimension}: "
                    f"{abs(comparison.effective_delta):.3f}."
                )

        if not verdict.blocked and not verdict.reasons:
            verdict.reasons.append("No significant regression on any guarded dimension.")
        return verdict


def _welch_p_value(
    baseline: QualityMeasurement, candidate: QualityMeasurement
) -> Decimal | None:
    """Two-sided Welch's t-test p-value, normal-approximated.

    Welch rather than Student's because the two samples routinely have
    different variances — a cheaper model is often not just worse on average
    but more erratic, and pooling the variance hides exactly that. The normal
    approximation is fine at our sample sizes (n >= 100 by policy) and avoids
    pulling in scipy for one function.
    """
    if baseline.sample_size < 2 or candidate.sample_size < 2:
        return None
    se_baseline = float(baseline.standard_error)
    se_candidate = float(candidate.standard_error)
    pooled = (se_baseline**2 + se_candidate**2) ** 0.5
    if pooled == 0:
        return None
    t_stat = abs(float(candidate.score) - float(baseline.score)) / pooled
    return to_decimal(2 * (1 - NormalDist().cdf(t_stat)))


# ---------------------------------------------------------------------------
# Deterministic scorers. Free, instant, and run on every request as a
# continuous signal rather than only during an experiment.
# ---------------------------------------------------------------------------


def groundedness_score(response: str, context_chunks: list[str], *, ngram: int = 8) -> Decimal:
    """Share of the response's n-grams traceable to retrieved context.

    A cheap proxy for "did the model make this up". Not a substitute for a real
    faithfulness evaluation — it rewards verbatim copying and penalises correct
    paraphrase — but it is free, runs on 100% of traffic, and reliably catches
    the specific failure that RAG context reduction causes: the model losing
    its evidence and falling back on parametric memory. Used as a *trend*
    signal; a sharp drop after a context change is the alarm.
    """
    if not response or not context_chunks:
        return ZERO
    haystack = " ".join(context_chunks).lower().split()
    if len(haystack) < ngram:
        return ZERO
    context_ngrams = {
        " ".join(haystack[i : i + ngram]) for i in range(len(haystack) - ngram + 1)
    }
    words = response.lower().split()
    if len(words) < ngram:
        return ZERO
    response_ngrams = [" ".join(words[i : i + ngram]) for i in range(len(words) - ngram + 1)]
    if not response_ngrams:
        return ZERO
    grounded = sum(1 for gram in response_ngrams if gram in context_ngrams)
    return safe_div(Decimal(grounded), Decimal(len(response_ngrams)))


def citation_score(response: str, *, expected_citations: int = 1) -> Decimal:
    """Presence of citation markers, for workloads that require sourcing."""
    import re

    markers = re.findall(r"\[\d+\]|\(source:[^)]+\)|\[\^?\w+\]", response)
    if expected_citations <= 0:
        return Decimal("1")
    return min(Decimal("1"), safe_div(Decimal(len(markers)), Decimal(expected_citations)))


def completeness_score(response: str, required_fields: list[str]) -> Decimal:
    """Share of required elements present — for structured extraction tasks."""
    if not required_fields:
        return Decimal("1")
    lowered = response.lower()
    present = sum(1 for field_name in required_fields if field_name.lower() in lowered)
    return safe_div(Decimal(present), Decimal(len(required_fields)))


def refusal_detected(response: str) -> bool:
    """Detect a model declining to answer.

    A spike in refusals after a model downgrade is a quality regression that
    accuracy scoring misses entirely — the model did not answer *wrongly*, it
    did not answer at all, and an accuracy metric computed over answered
    queries will happily report no change.
    """
    lowered = response.lower().strip()
    markers = (
        "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
        "i don't have access", "i do not have access", "i'm not able to",
        "cannot provide", "unable to assist",
    )
    return any(lowered.startswith(m) or f" {m}" in lowered[:400] for m in markers)


def build_baseline(
    subject: str, scores: dict[QualityDimension, list[Decimal]]
) -> QualityBaseline:
    """Aggregate per-request scores into a baseline with dispersion."""
    baseline = QualityBaseline(subject=subject)
    for dimension, values in scores.items():
        if not values:
            continue
        floats = [float(v) for v in values]
        mean = fmean(floats)
        variance = fmean([(v - mean) ** 2 for v in floats]) if len(floats) > 1 else 0.0
        baseline.measurements[dimension] = QualityMeasurement(
            dimension=dimension,
            score=to_decimal(mean),
            sample_size=len(values),
            stddev=to_decimal(variance**0.5),
        )
    return baseline
