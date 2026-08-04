/**
 * Executive dashboard.
 *
 * Audience: CFO, CTO, VP Engineering. They have about ninety seconds and one
 * question: is AI spend under control, and if not, what is being done about it.
 *
 * So the page answers in that order — spend and trajectory first, then the
 * identified opportunity with a dollar figure attached, then the exceptions
 * that need a decision. Detail is deliberately withheld; every tile links into
 * the operational dashboards for anyone who wants to go deeper.
 */

import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';

import { CostFlow, CostTrend, ForecastChart } from '../components/charts';
import {
  Badge,
  EmptyState,
  ErrorState,
  LoadingPanel,
  Panel,
  ProgressBar,
  SeverityChip,
  StatTile,
} from '../components/ui';
import { api, formatCurrency, formatPercent, toNumber } from '../lib/api';
import { budgetColor } from '../lib/theme';

const WINDOW_DAYS = 30;

export default function Executive() {
  const summary = useQuery({ queryKey: ['summary', WINDOW_DAYS], queryFn: () => api.summary(WINDOW_DAYS) });
  const series = useQuery({ queryKey: ['timeseries', WINDOW_DAYS], queryFn: () => api.timeseries(WINDOW_DAYS) });
  const forecast = useQuery({ queryKey: ['forecast', 30], queryFn: () => api.forecast(30) });
  const recommendations = useQuery({
    queryKey: ['recommendations', WINDOW_DAYS],
    queryFn: () => api.recommendations(WINDOW_DAYS),
  });
  const budgets = useQuery({ queryKey: ['budgets'], queryFn: () => api.budgets() });
  const anomalies = useQuery({ queryKey: ['anomalies', 7], queryFn: () => api.anomalies(7, 'medium') });
  const flow = useQuery({ queryKey: ['flow', WINDOW_DAYS], queryFn: () => api.flow(WINDOW_DAYS) });

  // Today is always partial — only the hours ingested so far are counted — so
  // it plots as a cliff toward zero at the right edge and reads as a spend
  // collapse. The backend excludes it from forecast fitting for the same
  // reason; the charts have to agree or the two panels contradict each other.
  const completeDays = (series.data ?? []).slice(0, -1);

  const opportunity = (recommendations.data ?? []).reduce(
    (sum, r) => sum + toNumber(r.estimated_monthly_savings),
    0,
  );
  const totalCost = toNumber(summary.data?.total_cost);
  const annualRunRate = toNumber(forecast.data?.horizon_totals?.['30d']) * 12;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold text-slate-50">Executive overview</h1>
        <p className="mt-1 text-sm text-slate-400">
          AI spend, trajectory and the identified savings opportunity across the last{' '}
          {WINDOW_DAYS} days.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile
          index={0}
          label="Spend (30d)"
          value={formatCurrency(totalCost, { compact: true })}
          delta={summary.data?.period_over_period_pct ? toNumber(summary.data.period_over_period_pct) : null}
          deltaLabel="vs prior 30d"
          hint="Resolved against effective-dated provider rate cards."
        />
        <StatTile
          index={1}
          label="Annualised run rate"
          value={formatCurrency(annualRunRate, { compact: true })}
          hint={
            forecast.data?.mape
              ? `Forecast MAPE ${formatPercent(forecast.data.mape)} on backtest.`
              : 'Projected from the 30-day forecast.'
          }
        />
        <StatTile
          index={2}
          label="Identified opportunity"
          value={`${formatCurrency(opportunity, { compact: true })}/mo`}
          hint={
            totalCost > 0
              ? `${formatPercent((opportunity / totalCost) * 100, 0)} of current spend, quality-gated.`
              : undefined
          }
          tone="positive"
        />
        <StatTile
          index={3}
          label="Wasted spend"
          value={formatCurrency(toNumber(summary.data?.wasted_cost), { compact: true })}
          hint="Failed, timed-out and retried requests that returned nothing."
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Panel
          className="xl:col-span-2"
          title="Spend and 30-day forecast"
          subtitle={
            forecast.data
              ? `${forecast.data.seasonal ? 'Weekly seasonality modelled' : 'Trend only'} · shaded band is the ${formatPercent(toNumber(forecast.data.confidence) * 100, 0)} confidence interval`
              : undefined
          }
        >
          {series.isLoading || forecast.isLoading ? (
            <LoadingPanel rows={5} />
          ) : series.isError ? (
            <ErrorState error={series.error} onRetry={() => series.refetch()} />
          ) : (
            <>
              <ForecastChart
                history={completeDays.slice(-45)}
                forecast={forecast.data?.points ?? []}
              />
              {(forecast.data?.warnings.length ?? 0) > 0 && (
                <ul className="mt-3 space-y-1">
                  {forecast.data!.warnings.map((warning) => (
                    <li key={warning} className="text-xs text-amber-300/80">
                      ⚠ {warning}
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </Panel>

        <Panel title="Budget health" subtitle="Current period utilisation">
          {budgets.isLoading ? (
            <LoadingPanel rows={3} />
          ) : budgets.isError ? (
            <ErrorState error={budgets.error} onRetry={() => budgets.refetch()} />
          ) : (budgets.data ?? []).length === 0 ? (
            <EmptyState
              title="No budgets configured"
              detail="Budgets turn cost visibility into cost control. Set one per department to enable forecasting alerts."
            />
          ) : (
            <ul className="space-y-4">
              {budgets.data!.map((budget) => {
                const utilisation = toNumber(budget.utilisation_pct);
                return (
                  <li key={budget.id}>
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="truncate text-xs font-medium text-slate-200">
                        {budget.name}
                      </span>
                      <span className="tabular shrink-0 text-xs text-slate-400">
                        {formatCurrency(budget.spent, { compact: true })} /{' '}
                        {formatCurrency(budget.amount, { compact: true })}
                      </span>
                    </div>
                    <div className="mt-1.5">
                      <ProgressBar
                        value={utilisation}
                        color={budgetColor(utilisation)}
                        label={`${budget.name} budget utilisation`}
                      />
                    </div>
                    <div className="mt-1 flex items-center justify-between">
                      <span className="tabular text-[11px] text-slate-500">
                        {formatPercent(utilisation)} used · {budget.period}
                      </span>
                      {budget.projected_to_exceed && <Badge tone="warning">Forecast to exceed</Badge>}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </Panel>
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Panel
          className="xl:col-span-2"
          title="Top savings opportunities"
          subtitle="Ranked by value per unit of engineering effort and risk"
          actions={
            <Link
              to="/optimization"
              className="focus-ring rounded-lg border border-white/10 px-2.5 py-1 text-xs text-slate-300 hover:bg-white/5"
            >
              View all
            </Link>
          }
        >
          {recommendations.isLoading ? (
            <LoadingPanel rows={4} />
          ) : recommendations.isError ? (
            <ErrorState error={recommendations.error} onRetry={() => recommendations.refetch()} />
          ) : (recommendations.data ?? []).length === 0 ? (
            <EmptyState
              title="No opportunities above the reporting threshold"
              detail="Findings worth under $10/month are suppressed so the list stays actionable."
            />
          ) : (
            <ul className="divide-y divide-white/5">
              {recommendations.data!.slice(0, 5).map((rec) => (
                <li key={rec.id} className="flex items-start justify-between gap-4 py-3 first:pt-0">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-slate-100">{rec.title}</p>
                    <p className="mt-0.5 line-clamp-2 text-xs leading-relaxed text-slate-400">
                      {rec.rationale}
                    </p>
                    <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                      <Badge tone={rec.effort === 'config_change' ? 'success' : 'neutral'}>
                        {rec.effort.replace(/_/g, ' ')}
                      </Badge>
                      {rec.requires_evaluation && <Badge tone="warning">Needs evaluation</Badge>}
                    </div>
                  </div>
                  <div className="shrink-0 text-right">
                    <p className="tabular text-sm font-semibold text-emerald-300">
                      {formatCurrency(rec.estimated_monthly_savings, { compact: true })}
                    </p>
                    <p className="text-[11px] text-slate-500">per month</p>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Needs attention" subtitle="Open anomalies, last 7 days">
          {anomalies.isLoading ? (
            <LoadingPanel rows={3} />
          ) : anomalies.isError ? (
            <ErrorState error={anomalies.error} onRetry={() => anomalies.refetch()} />
          ) : (anomalies.data ?? []).length === 0 ? (
            <EmptyState title="Nothing anomalous" detail="No cost, token or agent anomalies above medium severity." />
          ) : (
            <ul className="space-y-3">
              {anomalies.data!.slice(0, 5).map((anomaly, i) => (
                <li key={anomaly.id ?? i} className="rounded-lg border border-white/5 bg-white/[0.02] p-3">
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-xs font-medium text-slate-200">{anomaly.title}</p>
                    <SeverityChip severity={anomaly.severity} />
                  </div>
                  <p className="tabular mt-1.5 text-xs text-red-300">
                    {formatCurrency(anomaly.estimated_impact, { compact: true })} impact
                  </p>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel title="Where the money goes" subtitle="Department → team → provider → model">
          {flow.isLoading ? (
            <LoadingPanel rows={6} />
          ) : (flow.data ?? []).length === 0 ? (
            <EmptyState title="No attributed spend in this window" />
          ) : (
            <CostFlow links={flow.data!} />
          )}
        </Panel>

        <Panel title="Daily spend" subtitle={`Last ${WINDOW_DAYS} complete days`}>
          {series.isLoading ? (
            <LoadingPanel rows={5} />
          ) : (
            <CostTrend data={completeDays} />
          )}
        </Panel>
      </div>
    </div>
  );
}
