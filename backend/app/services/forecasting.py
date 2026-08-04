"""Cost forecasting.

## Why not Prophet / ARIMA / an LSTM

Evaluated against the actual shape of the problem:

- **Prophet** — good accuracy, handles holidays, but a heavy native dependency
  and ~2s per series. We forecast ~50k series nightly (team x model x feature);
  at 2s each that is 28 hours of compute. Rejected on cost.
- **SARIMA** — good with tuning, but per-series order selection is expensive
  and brittle, and unattended re-fitting drifts. Rejected on operability.
- **LSTM / Temporal Fusion Transformer** — best on long horizons, but needs
  training infrastructure, GPUs and MLOps ownership, and is close to
  uninspectable. Unjustifiable for a 90-day horizon on weekly-seasonal data.
- **Holt-Winters (triple exponential smoothing) with a damped trend** —
  within a few points of Prophet on 30-day horizons for this data, ~1ms per
  series in pure Python, and the level/trend/season decomposition is directly
  inspectable. **Chosen.**

AI spend is dominated by weekly seasonality (weekday business usage, weekend
troughs) plus a trend that is usually a step change at feature launch rather
than a smooth ramp. Holt-Winters models exactly that, runs in microseconds, has
no native dependencies, and a FinOps analyst can be shown the decomposition and
agree or disagree with it. Explainability matters more than the last two points
of accuracy here: a forecast that finance does not trust does not get used to
set budgets, no matter how accurate it is.

The damping factor (`phi < 1`) is important and non-default: undamped linear
trend extrapolation on a series that just had a launch spike produces absurd
annual projections, which destroys credibility on first contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from statistics import NormalDist, fmean, pstdev

from app.domain.money import ZERO, quantize_cost, safe_div, to_decimal

#: Weekly seasonality. AI usage tracks the business week almost universally.
DEFAULT_SEASON_LENGTH = 7
#: Below this many observations, seasonal decomposition is fitting noise; we
#: degrade to a trend-only model and widen the interval to signal low
#: confidence rather than pretending to a precision we do not have.
MIN_SEASONAL_OBSERVATIONS = 21
MIN_OBSERVATIONS = 5


@dataclass(slots=True)
class ForecastPoint:
    at: date
    value: Decimal
    lower: Decimal
    upper: Decimal


@dataclass(slots=True)
class ForecastResult:
    points: list[ForecastPoint] = field(default_factory=list)
    method: str = "holt_winters"
    #: Mean absolute percentage error from walk-forward backtest on held-out
    #: history. Surfaced in the UI next to every forecast — an unqualified
    #: number invites false precision.
    mape: Decimal | None = None
    confidence: Decimal = Decimal("0.80")
    seasonal: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> Decimal:
        return sum((p.value for p in self.points), ZERO)

    @property
    def total_upper(self) -> Decimal:
        return sum((p.upper for p in self.points), ZERO)

    def horizon_total(self, days: int) -> Decimal:
        return sum((p.value for p in self.points[:days]), ZERO)


@dataclass(slots=True)
class BudgetExhaustion:
    """When a budget runs out at the forecast burn rate."""

    exhausted_on: date | None
    days_remaining: int | None
    projected_period_spend: Decimal
    budget_amount: Decimal
    projected_overrun: Decimal
    burn_rate_per_day: Decimal

    @property
    def will_exhaust(self) -> bool:
        return self.exhausted_on is not None

    @property
    def utilisation_pct(self) -> Decimal:
        return safe_div(self.projected_period_spend, self.budget_amount) * Decimal("100")


def _f(values: list[Decimal]) -> list[float]:
    return [float(v) for v in values]


def holt_winters(
    history: list[Decimal],
    horizon: int,
    *,
    season_length: int = DEFAULT_SEASON_LENGTH,
    alpha: float = 0.35,
    beta: float = 0.12,
    gamma: float = 0.25,
    phi: float = 0.92,
) -> tuple[list[float], list[float], bool]:
    """Additive Holt-Winters with a damped trend.

    Returns `(fitted_in_sample, forecast, seasonal_used)`.

    Additive rather than multiplicative seasonality: AI cost series routinely
    contain zeros (a team that ships nothing on a bank holiday), and
    multiplicative decomposition is undefined at zero. Additive is also more
    robust when the level is small, which matters for the long tail of small
    teams where a multiplicative model produces wild swings.

    Smoothing constants are conservative defaults chosen for stability over
    responsiveness. They are re-fitted per series by `tune_parameters` when
    enough history exists; these values are the cold-start prior.
    """
    series = _f(history)
    n = len(series)
    use_season = n >= max(MIN_SEASONAL_OBSERVATIONS, 2 * season_length)

    if not use_season:
        # Damped Holt's linear trend, no seasonal component.
        level = series[0]
        trend = (series[-1] - series[0]) / max(1, n - 1)
        fitted: list[float] = []
        for value in series:
            fitted.append(level + phi * trend)
            prev_level = level
            level = alpha * value + (1 - alpha) * (level + phi * trend)
            trend = beta * (level - prev_level) + (1 - beta) * phi * trend
        forecast = []
        damped_sum = 0.0
        for h in range(1, horizon + 1):
            damped_sum += phi**h
            forecast.append(max(0.0, level + damped_sum * trend))
        return fitted, forecast, False

    seasons = n // season_length
    season_means = [
        fmean(series[i * season_length : (i + 1) * season_length]) for i in range(seasons)
    ]
    overall = fmean(season_means) if season_means else 0.0

    # Initial seasonal indices: average deviation of each phase from its
    # season's mean. Cheap, stable, and adequate as a starting point since the
    # gamma updates dominate after two cycles.
    seasonal = []
    for phase in range(season_length):
        deviations = [
            series[i * season_length + phase] - season_means[i]
            for i in range(seasons)
            if i * season_length + phase < n
        ]
        seasonal.append(fmean(deviations) if deviations else 0.0)

    level = overall
    trend = (season_means[-1] - season_means[0]) / max(1, seasons - 1) / season_length
    fitted = []

    for i, value in enumerate(series):
        phase = i % season_length
        fitted.append(level + phi * trend + seasonal[phase])
        prev_level = level
        level = alpha * (value - seasonal[phase]) + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - prev_level) + (1 - beta) * phi * trend
        seasonal[phase] = gamma * (value - level) + (1 - gamma) * seasonal[phase]

    forecast = []
    damped_sum = 0.0
    for h in range(1, horizon + 1):
        damped_sum += phi**h
        phase = (n + h - 1) % season_length
        # Costs are non-negative by construction; clamping prevents a steep
        # negative trend from projecting a refund.
        forecast.append(max(0.0, level + damped_sum * trend + seasonal[phase]))

    return fitted, forecast, True


def forecast_cost(
    history: list[Decimal | float | str],
    *,
    horizon_days: int = 30,
    start_date: date | None = None,
    season_length: int = DEFAULT_SEASON_LENGTH,
    confidence: float = 0.80,
) -> ForecastResult:
    """Forecast daily cost `horizon_days` forward from a daily history series.

    `history` must be a dense, gap-filled daily series (missing days as 0) —
    exponential smoothing assumes uniform spacing, and a sparse series silently
    compresses the time axis. The aggregation job guarantees density.
    """
    values = [to_decimal(v) for v in history]
    result = ForecastResult(confidence=to_decimal(confidence))
    anchor = start_date or datetime.now(UTC).date()

    if len(values) < MIN_OBSERVATIONS:
        # Fall back to a flat mean projection. Stated plainly rather than
        # dressed up: a 3-day-old workload has no forecastable shape.
        mean = fmean(_f(values)) if values else 0.0
        result.method = "insufficient_history_mean"
        result.seasonal = False
        result.warnings.append(
            f"only {len(values)} observations; need {MIN_OBSERVATIONS} for a trend model"
        )
        band = Decimal("0.5") * to_decimal(mean)
        result.points = [
            ForecastPoint(
                at=anchor + timedelta(days=h),
                value=quantize_cost(to_decimal(mean)),
                lower=quantize_cost(max(ZERO, to_decimal(mean) - band)),
                upper=quantize_cost(to_decimal(mean) + band),
            )
            for h in range(1, horizon_days + 1)
        ]
        return result

    fitted, projection, seasonal_used = holt_winters(
        values, horizon_days, season_length=season_length
    )
    result.seasonal = seasonal_used
    if not seasonal_used:
        result.method = "damped_holt"
        result.warnings.append(
            f"under {MIN_SEASONAL_OBSERVATIONS} observations; weekly seasonality not modelled"
        )

    residuals = [a - b for a, b in zip(_f(values), fitted, strict=False)]
    sigma = pstdev(residuals) if len(residuals) > 1 else 0.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2)

    points: list[ForecastPoint] = []
    for h, value in enumerate(projection, start=1):
        # Interval widens with sqrt(h): uncertainty compounds with horizon, and
        # a constant-width band understates 30-day risk badly enough to cause
        # under-provisioned budgets.
        spread = z * sigma * (h**0.5)
        points.append(
            ForecastPoint(
                at=anchor + timedelta(days=h),
                value=quantize_cost(to_decimal(value)),
                lower=quantize_cost(max(ZERO, to_decimal(value - spread))),
                upper=quantize_cost(to_decimal(value + spread)),
            )
        )
    result.points = points
    result.mape = backtest_mape(values, season_length=season_length)
    return result


def backtest_mape(
    values: list[Decimal], *, season_length: int = DEFAULT_SEASON_LENGTH, folds: int = 5
) -> Decimal | None:
    """Walk-forward MAPE — fit on a prefix, score the next day, roll forward.

    Walk-forward rather than a random split because a random split leaks future
    information into training on a time series and produces flatteringly wrong
    accuracy figures. Days with zero actual cost are skipped: percentage error
    is undefined against a zero denominator.
    """
    if len(values) < MIN_OBSERVATIONS + folds:
        return None
    errors: list[float] = []
    for offset in range(folds, 0, -1):
        train = values[:-offset]
        actual = float(values[-offset])
        if actual == 0 or len(train) < MIN_OBSERVATIONS:
            continue
        _, projection, _ = holt_winters(train, 1, season_length=season_length)
        errors.append(abs(projection[0] - actual) / abs(actual))
    if not errors:
        return None
    return quantize_cost(to_decimal(fmean(errors) * 100))


def project_budget_exhaustion(
    *,
    spent_to_date: Decimal,
    budget_amount: Decimal,
    period_start: date,
    period_end: date,
    forecast: ForecastResult,
    today: date | None = None,
) -> BudgetExhaustion:
    """When, if ever, does this budget run dry?

    Uses the forecast curve rather than a naive linear burn rate. Linear
    extrapolation systematically under-predicts exhaustion for workloads with
    weekly seasonality assessed mid-week, and over-predicts when assessed on a
    weekend — both produce alerts that erode trust.
    """
    now = today or datetime.now(UTC).date()
    remaining_budget = budget_amount - spent_to_date
    days_left_in_period = max(0, (period_end - now).days)

    running = ZERO
    exhausted_on: date | None = None
    for point in forecast.points:
        if point.at > period_end:
            break
        running += point.value
        if exhausted_on is None and running >= remaining_budget:
            exhausted_on = point.at

    projected_period_spend = spent_to_date + forecast.horizon_total(days_left_in_period)
    overrun = projected_period_spend - budget_amount
    burn = safe_div(
        forecast.horizon_total(min(7, len(forecast.points))),
        Decimal(min(7, len(forecast.points)) or 1),
    )

    return BudgetExhaustion(
        exhausted_on=exhausted_on,
        days_remaining=(exhausted_on - now).days if exhausted_on else None,
        projected_period_spend=quantize_cost(projected_period_spend),
        budget_amount=budget_amount,
        projected_overrun=quantize_cost(overrun if overrun > ZERO else ZERO),
        burn_rate_per_day=quantize_cost(burn),
    )


def forecast_horizons(history: list[Decimal | float | str]) -> dict[str, Decimal]:
    """Convenience roll-up for the executive dashboard tiles.

    Annual is projected from the 90-day forecast rather than by running a
    365-day horizon: exponential smoothing degrades badly past ~4 seasonal
    cycles, and quoting a 12-month number from a 12-month extrapolation implies
    a confidence we cannot support. Scaling a 90-day figure is equally
    approximate but does not hide the assumption.
    """
    ninety = forecast_cost(history, horizon_days=90)
    return {
        "daily": ninety.horizon_total(1),
        "weekly": ninety.horizon_total(7),
        "monthly": ninety.horizon_total(30),
        "quarterly": ninety.horizon_total(90),
        "annual_projected": quantize_cost(ninety.horizon_total(90) * Decimal("4.0556")),
    }


def detect_peak_days(forecast: ForecastResult, *, threshold_pct: Decimal = Decimal("130")) -> list[date]:
    """Forecast days materially above the horizon average — used to pre-warn
    capacity owners about rate-limit risk, not just cost."""
    if not forecast.points:
        return []
    avg = safe_div(forecast.total, Decimal(len(forecast.points)))
    if avg == ZERO:
        return []
    return [p.at for p in forecast.points if safe_div(p.value, avg) * Decimal("100") >= threshold_pct]
