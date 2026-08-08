/**
 * Engineering dashboard.
 *
 * Audience: the engineers who own the prompts and pipelines. Their question is
 * not "how much did we spend" but "which part of my prompt is expensive, and
 * what do I change".
 *
 * So this page leads with prompt decomposition and the context-growth slope —
 * the two numbers that convert a cost conversation into a code change.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import { BreakdownBars, CompositionDonut, CostHeatmap, TokenBars } from '../components/charts';
import {
  EmptyState,
  ErrorState,
  LoadingPanel,
  Panel,
  Select,
  StatTile,
} from '../components/ui';
import { api, formatCurrency, formatNumber, formatPercent, toNumber } from '../lib/api';

const DIMENSIONS = [
  { value: 'model', label: 'Model' },
  { value: 'feature', label: 'Feature' },
  { value: 'team', label: 'Team' },
  { value: 'provider', label: 'Provider' },
  { value: 'application', label: 'Application' },
  { value: 'status', label: 'Status' },
];

export default function Engineering() {
  const [days, setDays] = useState(30);
  const [dimension, setDimension] = useState('feature');

  const tokens = useQuery({ queryKey: ['tokens', days], queryFn: () => api.tokens(days) });
  const breakdown = useQuery({
    queryKey: ['breakdown', dimension, days],
    queryFn: () => api.breakdown(dimension, days),
  });
  const heatmap = useQuery({
    queryKey: ['heatmap', days],
    queryFn: () => api.heatmap('team', 'model', days),
  });
  const summary = useQuery({ queryKey: ['summary', days], queryFn: () => api.summary(days) });

  const growth = toNumber(tokens.data?.context_growth_per_turn);
  const staticShare = toNumber(tokens.data?.static_share_pct);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-50">Engineering</h1>
          <p className="mt-1 text-sm text-slate-400">
            Where tokens go inside your prompts, and which workloads drive them.
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
          label="Prompt size (p50)"
          value={formatNumber(tokens.data?.prompt_p50 ?? 0, { compact: true })}
          hint={`p95 ${formatNumber(tokens.data?.prompt_p95 ?? 0, { compact: true })} · p99 ${formatNumber(tokens.data?.prompt_p99 ?? 0, { compact: true })} tokens`}
        />
        <StatTile
          index={1}
          label="Static prompt share"
          value={formatPercent(staticShare, 0)}
          hint={
            staticShare > 40
              ? 'High — this portion is identical on every call and is a strong prompt-cache candidate.'
              : 'Portion of the prompt that never changes between calls.'
          }
        />
        <StatTile
          index={2}
          label="Context growth per turn"
          value={`+${formatNumber(growth)} tok`}
          tone={growth > 300 ? 'negative' : 'neutral'}
          hint={
            growth > 300
              ? 'Steep — history is being resent verbatim, so cost grows quadratically with conversation length.'
              : 'Additional prompt tokens for each extra conversation turn.'
          }
        />
        <StatTile
          index={3}
          label="Cost per 1k tokens"
          value={formatCurrency(summary.data?.cost_per_1k_tokens ?? 0)}
          hint="Blended across every model and token class in this window."
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel
          title="Prompt composition"
          subtitle="Average share of input tokens by region"
        >
          {tokens.isLoading ? (
            <LoadingPanel rows={5} />
          ) : tokens.isError ? (
            <ErrorState error={tokens.error} onRetry={() => tokens.refetch()} />
          ) : (
            <>
              <CompositionDonut composition={tokens.data?.composition ?? {}} />
              <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                System prompt, tool definitions and few-shot examples are call-invariant. If they
                dominate this chart, provider prompt caching bills them at roughly a tenth of the
                input rate with no change to model behaviour.
              </p>
            </>
          )}
        </Panel>

        <Panel
          title="Cost by dimension"
          subtitle="Ranked spend"
          actions={
            <Select
              label="Group by"
              value={dimension}
              onChange={setDimension}
              options={DIMENSIONS}
            />
          }
        >
          {breakdown.isLoading ? (
            <LoadingPanel rows={5} />
          ) : breakdown.isError ? (
            <ErrorState error={breakdown.error} onRetry={() => breakdown.refetch()} />
          ) : (breakdown.data ?? []).length === 0 ? (
            <EmptyState title="No spend in this window" />
          ) : (
            <BreakdownBars data={breakdown.data!} />
          )}
        </Panel>
      </div>

      <Panel title="Team × model spend" subtitle="Which teams use which models, and at what cost">
        {heatmap.isLoading ? (
          <LoadingPanel rows={6} />
        ) : heatmap.isError ? (
          <ErrorState error={heatmap.error} onRetry={() => heatmap.refetch()} />
        ) : (heatmap.data?.rows ?? []).length === 0 ? (
          <EmptyState title="No attributed usage" detail="Instrument the SDK with team ids to populate this view." />
        ) : (
          <CostHeatmap
            rows={heatmap.data!.rows}
            columns={heatmap.data!.columns}
            values={heatmap.data!.values}
          />
        )}
      </Panel>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel title="Token volume" subtitle={`By ${dimension}`}>
          {breakdown.isLoading ? (
            <LoadingPanel rows={4} />
          ) : (
            <TokenBars data={breakdown.data ?? []} />
          )}
        </Panel>

        <Panel title="Efficiency detail" subtitle="Per-slice cost and latency">
          {breakdown.isLoading ? (
            <LoadingPanel rows={5} />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-slate-400">
                  <tr className="border-b border-white/5">
                    <th className="pb-2 pr-3 font-medium">Name</th>
                    <th className="pb-2 pr-3 text-right font-medium">Requests</th>
                    <th className="pb-2 pr-3 text-right font-medium">$/request</th>
                    <th className="pb-2 pr-3 text-right font-medium">Latency</th>
                    <th className="pb-2 text-right font-medium">Share</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {(breakdown.data ?? []).slice(0, 10).map((row) => (
                    <tr key={row.key}>
                      <td className="max-w-[12rem] truncate py-2 pr-3 text-slate-200" title={row.key}>
                        {row.key}
                      </td>
                      <td className="tabular py-2 pr-3 text-right text-slate-400">
                        {formatNumber(row.requests, { compact: true })}
                      </td>
                      <td className="tabular py-2 pr-3 text-right text-slate-300">
                        {formatCurrency(row.avg_cost_per_request)}
                      </td>
                      <td className="tabular py-2 pr-3 text-right text-slate-400">
                        {formatNumber(toNumber(row.avg_latency_ms))} ms
                      </td>
                      <td className="tabular py-2 text-right text-slate-300">
                        {formatPercent(row.share_pct)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
