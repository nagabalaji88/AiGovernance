"""Money handling.

Design decision: all monetary values are `decimal.Decimal`, never `float`.

Rationale — a single request can cost fractions of a cent, but a Fortune 500
tenant aggregates hundreds of millions of them per month into invoices that
must reconcile against provider bills to the cent. Binary floating point
accumulates representation error under summation; at 10^8 additions the drift
is material and, worse, non-deterministic across shard orderings, so two
recomputations of the same month disagree. Decimal is exact for the base-10
quantities we bill in.

Storage: `NUMERIC(20, 10)` in Postgres. Ten fractional digits holds a
per-token rate for the cheapest models (~$0.0000001/token) without truncation.
Presentation rounds to 2dp only at the very edge (invoice, UI), never in
intermediate aggregation.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Final

#: Precision retained for stored/aggregated amounts.
COST_EXPONENT: Final[Decimal] = Decimal("0.0000000001")
#: Precision used when presenting an amount to a human or an invoice line.
DISPLAY_EXPONENT: Final[Decimal] = Decimal("0.01")

ZERO: Final[Decimal] = Decimal("0")


def to_decimal(value: Decimal | int | float | str | None, *, default: Decimal = ZERO) -> Decimal:
    """Coerce arbitrary numeric input to Decimal without float contamination.

    Floats are routed through `str()` deliberately: `Decimal(0.1)` yields the
    full binary expansion (0.1000000000000000055511...), whereas
    `Decimal(str(0.1))` yields exactly `0.1`, which is what the caller meant.
    """
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def quantize_cost(value: Decimal) -> Decimal:
    """Round to storage precision. Applied once, at the persistence boundary."""
    return value.quantize(COST_EXPONENT, rounding=ROUND_HALF_UP)


def quantize_display(value: Decimal) -> Decimal:
    """Round to currency presentation precision (2dp)."""
    return value.quantize(DISPLAY_EXPONENT, rounding=ROUND_HALF_UP)


def safe_div(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Division that yields 0 rather than raising on a zero denominator.

    Ratio metrics (cost-per-token, savings percentage, cache hit rate) are
    computed over windows that are legitimately empty — a team with no traffic
    yesterday is not an error condition, and a dashboard tile should render
    `0`, not a 500.
    """
    if denominator == 0:
        return ZERO
    return numerator / denominator


def pct_change(current: Decimal, baseline: Decimal) -> Decimal:
    """Percentage change from baseline to current, as a 0-100 style figure.

    Returns 0 when the baseline is 0 — an increase from nothing is
    mathematically infinite but operationally meaningless as a percentage; the
    anomaly detector uses absolute thresholds for that case instead.
    """
    if baseline == 0:
        return ZERO
    return ((current - baseline) / baseline) * Decimal("100")
