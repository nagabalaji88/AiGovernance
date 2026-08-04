/**
 * Governance dashboard — budgets, chargeback and anomalies.
 *
 * Audience: FinOps. The distinguishing need here is *defensibility*. Every
 * number on this page may end up in a chargeback statement that a department
 * head disputes, so each is shown with its basis: what was directly attributed,
 * what share of unattributable cost was allocated, and on what proportion.
 * A single blended figure invites an argument nobody can settle.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import {
  Badge,
  EmptyState,
  ErrorState,
  LoadingPanel,
  Panel,
  ProgressBar,
  SeverityChip,
  Select,
  StatTile,
} from '../components/ui';
import { api, formatCurrency, formatNumber, formatPercent, toNumber } from '../lib/api';
import { budgetColor } from '../lib/theme';

export default function Governance() {
  const [days, setDays] = useState(30);

  const budgets = useQuery({ queryKey: ['budgets'], queryFn: () => api.budgets() });
  const chargeback = useQuery({ queryKey: ['chargeback', days], queryFn: () => api.chargeback(days) });
  const anomalies = useQuery({ queryKey: ['anomalies', days], queryFn: () => api.anomalies(days) });
  const summary = useQuery({ queryKey: ['summary', days], queryFn: () => api.summary(days) });

  const totalAnomalyImpact = (anomalies.data ?? []).reduce(
    (sum, a) => sum + toNumber(a.estimated_impact),
    0,
  );
  const coverage = toNumber(summary.data?.attribution_coverage_pct);
  const exceeded = (budgets.data ?? []).filter((b) => b.is_exceeded).length;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-50">Governance</h1>
          <p className="mt-1 text-sm text-slate-400">
            Budgets, chargeback and the exceptions that need a decision.
          </p>
        </div>
        <Select
          label="Window"
          value={String(days)}
          onChange={(v) => setDays(Number(v))}
          options={[
            { value: '7', label: 'Last 7 days' },
            { value: '30', label: 'Last 30 days' },
            { value: '90', label: 'Last 90 days' },
          ]}
        />
      </header>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile
          index={0}
          label="Attribution coverage"
          value={formatPercent(coverage, 0)}
          tone="positive"
          hint={
            coverage < 95
              ? 'Unattributed spend cannot be charged back. Instrument the remaining callers with cost-centre tags.'
              : 'Share of spend carrying a cost centre.'
          }
        />
        <StatTile
          index={1}
          label="Budgets exceeded"
          value={String(exceeded)}
          hint={`${(budgets.data ?? []).length} budgets configured.`}
        />
        <StatTile
          index={2}
          label="Anomaly impact"
          value={formatCurrency(totalAnomalyImpact, { compact: true })}
          hint={`Across ${(anomalies.data ?? []).length} open findings in this window.`}
        />
        <StatTile
          index={3}
          label="Wasted spend"
          value={formatCurrency(toNumber(summary.data?.wasted_cost), { compact: true })}
          hint={`${formatPercent(toNumber(summary.data?.waste_ratio_pct))} of total — failures and retries.`}
        />
      </div>

      <Panel title="Budgets" subtitle="Current period utilisation and enforcement">
        {budgets.isLoading ? (
          <LoadingPanel rows={3} />
        ) : budgets.isError ? (
          <ErrorState error={budgets.error} onRetry={() => budgets.refetch()} />
        ) : (budgets.data ?? []).length === 0 ? (
          <EmptyState
            title="No budgets configured"
            detail="Without a budget there is nothing to enforce against and no forecast-based early warning."
          />
        ) : (
          <ul className="grid gap-4 md:grid-cols-2">
            {budgets.data!.map((budget) => {
              const utilisation = toNumber(budget.utilisation_pct);
              return (
                <li key={budget.id} className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-slate-100">{budget.name}</p>
                      <p className="mt-0.5 text-[11px] capitalize text-slate-500">
                        {budget.scope} · {budget.period} · {budget.period_start} to{' '}
                        {budget.period_end}
                      </p>
                    </div>
                    <SeverityChip severity={budget.severity} />
                  </div>
                  <div className="mt-3">
                    <ProgressBar
                      value={utilisation}
                      color={budgetColor(utilisation)}
                      label={`${budget.name} utilisation`}
                    />
                  </div>
                  <div className="tabular mt-2 flex items-center justify-between text-xs">
                    <span className="text-slate-300">
                      {formatCurrency(budget.spent, { compact: true })} of{' '}
                      {formatCurrency(budget.amount, { compact: true })}
                    </span>
                    <span className="text-slate-500">{formatPercent(utilisation)}</span>
                  </div>
                  {budget.projected_to_exceed && (
                    <p className="mt-2">
                      <Badge tone="warning">Forecast to exceed this period</Badge>
                    </p>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </Panel>

      <Panel
        title="Chargeback"
        subtitle="Direct cost plus a proportional share of unattributable spend"
      >
        {chargeback.isLoading ? (
          <LoadingPanel rows={5} />
        ) : chargeback.isError ? (
          <ErrorState error={chargeback.error} onRetry={() => chargeback.refetch()} />
        ) : (chargeback.data ?? []).length === 0 ? (
          <EmptyState
            title="No attributed spend"
            detail="Chargeback requires cost-centre tags on usage events."
          />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-slate-400">
                  <tr className="border-b border-white/5">
                    <th className="pb-2 pr-3 font-medium">Cost centre</th>
                    <th className="pb-2 pr-3 text-right font-medium">Direct</th>
                    <th className="pb-2 pr-3 text-right font-medium">Allocated shared</th>
                    <th className="pb-2 pr-3 text-right font-medium">Total</th>
                    <th className="pb-2 pr-3 text-right font-medium">Requests</th>
                    <th className="pb-2 text-right font-medium">Share</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {chargeback.data!.map((line) => (
                    <tr key={line.cost_center}>
                      <td className="py-2 pr-3 font-medium text-slate-200">{line.cost_center}</td>
                      <td className="tabular py-2 pr-3 text-right text-slate-300">
                        {formatCurrency(line.direct_cost, { compact: true })}
                      </td>
                      <td className="tabular py-2 pr-3 text-right text-slate-400">
                        {formatCurrency(line.allocated_shared_cost, { compact: true })}
                      </td>
                      <td className="tabular py-2 pr-3 text-right font-semibold text-slate-100">
                        {formatCurrency(line.total, { compact: true })}
                      </td>
                      <td className="tabular py-2 pr-3 text-right text-slate-400">
                        {formatNumber(line.requests, { compact: true })}
                      </td>
                      <td className="tabular py-2 text-right text-slate-300">
                        {formatPercent(line.share_pct)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
              Unattributable spend is allocated proportionally to each cost centre's direct usage
              rather than split evenly — the method finance already applies to shared
              infrastructure. An even split penalises small teams and reliably starts turf wars.
            </p>
          </>
        )}
      </Panel>

      <Panel title="Anomalies" subtitle="Ranked by dollar impact, most expensive first">
        {anomalies.isLoading ? (
          <LoadingPanel rows={4} />
        ) : anomalies.isError ? (
          <ErrorState error={anomalies.error} onRetry={() => anomalies.refetch()} />
        ) : (anomalies.data ?? []).length === 0 ? (
          <EmptyState
            title="No anomalies detected"
            detail="No cost spikes, retry storms, runaway agents or context explosions in this window."
          />
        ) : (
          <ul className="space-y-3">
            {anomalies.data!.map((anomaly, i) => (
              <li
                key={anomaly.id ?? i}
                className="rounded-xl border border-white/5 bg-white/[0.02] p-4"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="text-sm font-medium text-slate-100">{anomaly.title}</p>
                      <SeverityChip severity={anomaly.severity} />
                      <Badge>{anomaly.kind.replace(/_/g, ' ')}</Badge>
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-slate-400">{anomaly.detail}</p>
                    {anomaly.recommended_action && (
                      <p className="mt-2 text-xs leading-relaxed text-indigo-300/80">
                        → {anomaly.recommended_action}
                      </p>
                    )}
                  </div>
                  <div className="shrink-0 text-right">
                    <p className="tabular text-sm font-semibold text-red-300">
                      {formatCurrency(anomaly.estimated_impact, { compact: true })}
                    </p>
                    <p className="text-[11px] text-slate-500">estimated impact</p>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
