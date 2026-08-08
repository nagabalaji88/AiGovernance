"""Forecasting behaviour, including the failure modes we deliberately guard."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.services.forecasting import (
    MIN_OBSERVATIONS,
    backtest_mape,
    detect_peak_days,
    forecast_cost,
    forecast_horizons,
    holt_winters,
    project_budget_exhaustion,
)


def weekly_series(
    weeks: int, *, base: float = 100.0, trend: float = 0.0, noise: float = 0.0
) -> list[Decimal]:
    """Weekday-heavy series with a weekend trough — the real shape of AI spend.

    `noise` uses a fixed-seed generator so tests stay deterministic. Real cost
    series are never noiseless, and a perfectly smooth series produces zero
    residuals and therefore zero-width prediction intervals — correct, but not
    representative of anything the forecaster will actually meet.
    """
    import random

    rng = random.Random(1234)
    shape = [1.0, 1.05, 1.1, 1.05, 0.95, 0.25, 0.2]
    return [
        Decimal(str(round(base * shape[d % 7] + trend * d + rng.uniform(-noise, noise), 4)))
        for d in range(weeks * 7)
    ]


class TestHoltWinters:
    def test_learns_weekly_seasonality(self) -> None:
        history = weekly_series(8)
        _, projection, seasonal = holt_winters(history, 7)
        assert seasonal
        # The forecast's weekend days must be materially below its weekdays.
        weekday_avg = sum(projection[:5]) / 5
        weekend_avg = sum(projection[5:7]) / 2
        assert weekend_avg < weekday_avg * 0.5

    def test_falls_back_to_trend_only_on_short_history(self) -> None:
        _, _, seasonal = holt_winters([Decimal("10")] * 10, 5)
        assert not seasonal

    def test_never_projects_negative_cost(self) -> None:
        """A steep decline must floor at zero, not project a refund."""
        declining = [Decimal(str(max(0, 500 - i * 25))) for i in range(21)]
        _, projection, _ = holt_winters(declining, 30)
        assert all(v >= 0 for v in projection)

    def test_damping_prevents_runaway_extrapolation(self) -> None:
        """Undamped linear trend on a launch spike yields absurd annual figures."""
        spiking = [Decimal(str(10 * (i + 1))) for i in range(30)]
        _, projection, _ = holt_winters(spiking, 90)
        undamped_final = 10 * 30 + 10 * 90
        assert projection[-1] < undamped_final


class TestForecastCost:
    def test_produces_requested_horizon(self) -> None:
        result = forecast_cost(weekly_series(6), horizon_days=45)
        assert len(result.points) == 45

    def test_intervals_widen_with_horizon(self) -> None:
        """Constant-width bands understate 30-day risk and under-provision budgets."""
        result = forecast_cost(weekly_series(8, noise=8.0), horizon_days=30)
        first = result.points[0].upper - result.points[0].lower
        last = result.points[-1].upper - result.points[-1].lower
        assert first > 0
        assert last > first

    def test_lower_bound_never_negative(self) -> None:
        result = forecast_cost(weekly_series(6, base=5), horizon_days=30)
        assert all(p.lower >= 0 for p in result.points)

    def test_insufficient_history_says_so_rather_than_guessing(self) -> None:
        result = forecast_cost([Decimal("10")] * 3, horizon_days=7)
        assert result.method == "insufficient_history_mean"
        assert result.warnings
        assert len(result.points) == 7

    def test_short_history_warns_about_missing_seasonality(self) -> None:
        result = forecast_cost([Decimal("10")] * (MIN_OBSERVATIONS + 2), horizon_days=7)
        assert not result.seasonal
        assert any("seasonality" in w for w in result.warnings)

    def test_reports_backtested_accuracy(self) -> None:
        """An unqualified forecast invites false precision."""
        result = forecast_cost(weekly_series(10), horizon_days=14)
        assert result.mape is not None
        assert result.mape >= 0

    def test_tracks_an_upward_trend(self) -> None:
        result = forecast_cost(weekly_series(10, trend=2.0), horizon_days=14)
        history_avg = sum(weekly_series(10, trend=2.0)[-7:]) / 7
        forecast_avg = result.total / Decimal("14")
        assert forecast_avg > history_avg * Decimal("0.9")


class TestBacktest:
    def test_returns_none_without_enough_history(self) -> None:
        assert backtest_mape([Decimal("1")] * 4) is None

    def test_perfectly_flat_series_has_near_zero_error(self) -> None:
        mape = backtest_mape([Decimal("100")] * 40)
        assert mape is not None
        assert mape < Decimal("5")


class TestBudgetExhaustion:
    def test_predicts_exhaustion_date_for_an_overspending_budget(self) -> None:
        forecast = forecast_cost([Decimal("100")] * 30, horizon_days=30)
        today = date(2026, 8, 4)
        result = project_budget_exhaustion(
            spent_to_date=Decimal("900"),
            budget_amount=Decimal("1000"),
            period_start=date(2026, 8, 1),
            period_end=date(2026, 8, 31),
            forecast=forecast,
            today=today,
        )
        assert result.will_exhaust
        assert result.exhausted_on is not None
        assert result.days_remaining is not None and result.days_remaining <= 3

    def test_healthy_budget_reports_no_exhaustion(self) -> None:
        forecast = forecast_cost([Decimal("10")] * 30, horizon_days=30)
        result = project_budget_exhaustion(
            spent_to_date=Decimal("100"),
            budget_amount=Decimal("100000"),
            period_start=date(2026, 8, 1),
            period_end=date(2026, 8, 31),
            forecast=forecast,
            today=date(2026, 8, 4),
        )
        assert not result.will_exhaust
        assert result.projected_overrun == Decimal("0")

    def test_projected_overrun_is_reported(self) -> None:
        forecast = forecast_cost([Decimal("500")] * 30, horizon_days=30)
        result = project_budget_exhaustion(
            spent_to_date=Decimal("2000"),
            budget_amount=Decimal("3000"),
            period_start=date(2026, 8, 1),
            period_end=date(2026, 8, 31),
            forecast=forecast,
            today=date(2026, 8, 4),
        )
        assert result.projected_overrun > Decimal("0")
        assert result.utilisation_pct > Decimal("100")


class TestHorizons:
    def test_all_horizons_are_ordered(self) -> None:
        horizons = forecast_horizons(weekly_series(12))
        assert horizons["daily"] <= horizons["weekly"] <= horizons["monthly"]
        assert horizons["monthly"] <= horizons["quarterly"]
        assert horizons["annual_projected"] > horizons["quarterly"]


class TestPeakDetection:
    def test_identifies_days_above_the_horizon_average(self) -> None:
        forecast = forecast_cost(weekly_series(10), horizon_days=28)
        peaks = detect_peak_days(forecast, threshold_pct=Decimal("120"))
        assert peaks
        assert all(isinstance(p, date) for p in peaks)

    def test_flat_series_has_no_peaks(self) -> None:
        forecast = forecast_cost([Decimal("100")] * 40, horizon_days=14)
        assert detect_peak_days(forecast, threshold_pct=Decimal("150")) == []
