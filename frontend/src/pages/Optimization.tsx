/**
 * Optimization dashboard.
 *
 * The product's action surface. Two deliberate choices:
 *
 * 1. **Quick wins are separated out.** Config-only, zero-risk, no-evaluation
 *    changes are shown first and separately. Teams that start with a
 *    three-week migration abandon the programme; teams that start with a
 *    one-line change that saves real money come back for the hard ones.
 *
 * 2. **Quality-blocked recommendations are shown, not hidden.** If the gate
 *    refuses an otherwise attractive saving, the user sees it *and the reason*.
 *    Silently omitting them leads teams to implement the change themselves
 *    without the gate, which is the worst outcome available.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import { SavingsWaterfall } from '../components/charts';
import {
  Badge,
  EmptyState,
  ErrorState,
  LoadingPanel,
  Panel,
  StatTile,
} from '../components/ui';
import { api, formatCurrency, formatPercent, toNumber, type Recommendation } from '../lib/api';
import { RISK_STYLES } from '../lib/theme';

export default function Optimization() {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [showBlocked, setShowBlocked] = useState(false);

  const recommendations = useQuery({
    queryKey: ['recommendations', 30, showBlocked],
    queryFn: () => api.recommendations(30, showBlocked),
  });
  const summary = useQuery({ queryKey: ['summary', 30], queryFn: () => api.summary(30) });

  const all = recommendations.data ?? [];
  const actionable = all.filter((r) => !r.blocked_by_quality);
  const blocked = all.filter((r) => r.blocked_by_quality);
  const quickWins = actionable.filter(
    (r) => r.effort === 'config_change' && !r.requires_evaluation && ['none', 'low'].includes(r.risk),
  );

  const monthlyOpportunity = actionable.reduce(
    (sum, r) => sum + toNumber(r.estimated_monthly_savings),
    0,
  );
  const quickWinTotal = quickWins.reduce((sum, r) => sum + toNumber(r.estimated_monthly_savings), 0);
  const currentSpend = toNumber(summary.data?.total_cost);

  // Label bars by scope, not by kind. Several recommendations share a kind
  // ("prompt_cache" applies to every model with a static prefix), so labelling
  // by kind produces an axis reading "prompt cache, prompt cache, semantic
  // cache, semantic cache" where no bar can be matched to its recommendation.
  const waterfallItems = actionable.slice(0, 6).map((r) => ({
    label: r.scope_key.split(':').pop()?.split('/').pop() ?? r.kind,
    amount: toNumber(r.estimated_monthly_savings),
  }));

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold text-slate-50">Optimization</h1>
        <p className="mt-1 text-sm text-slate-400">
          Every identified saving, ranked by value per unit of engineering effort and risk.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile
          index={0}
          label="Monthly opportunity"
          value={formatCurrency(monthlyOpportunity, { compact: true })}
          tone="positive"
          hint={
            currentSpend > 0
              ? `${formatPercent((monthlyOpportunity / currentSpend) * 100, 0)} of current spend`
              : undefined
          }
        />
        <StatTile
          index={1}
          label="Annualised"
          value={formatCurrency(monthlyOpportunity * 12, { compact: true })}
          tone="positive"
          hint="Excludes anything the quality gate has blocked."
        />
        <StatTile
          index={2}
          label="Quick wins"
          value={formatCurrency(quickWinTotal, { compact: true })}
          tone="positive"
          hint={`${quickWins.length} config-only changes, no evaluation needed.`}
        />
        <StatTile
          index={3}
          label="Blocked on quality"
          value={String(blocked.length)}
          hint="Savings the gate refused because a guarded quality dimension regressed."
        />
      </div>

      {quickWins.length > 0 && (
        <Panel
          title="Start here — quick wins"
          subtitle="Configuration-only, no measurable quality risk, no evaluation run required"
        >
          <ul className="grid gap-3 md:grid-cols-2">
            {quickWins.map((rec) => (
              <li key={rec.id} className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.04] p-4">
                <div className="flex items-start justify-between gap-3">
                  <p className="text-sm font-medium text-slate-100">{rec.title}</p>
                  <span className="tabular shrink-0 text-sm font-semibold text-emerald-300">
                    {formatCurrency(rec.estimated_monthly_savings, { compact: true })}
                  </span>
                </div>
                <p className="mt-1.5 text-xs leading-relaxed text-slate-400">{rec.rationale}</p>
                <p className="mt-2 text-[11px] text-slate-500">
                  ~{toNumber(rec.implementation_hours)}h to implement · confidence{' '}
                  {formatPercent(toNumber(rec.confidence) * 100, 0)}
                </p>
              </li>
            ))}
          </ul>
        </Panel>
      )}

      <div className="grid gap-6 xl:grid-cols-3">
        <Panel
          className="xl:col-span-2"
          title="All recommendations"
          subtitle="Ranked by priority score = annual savings × confidence ÷ (effort × risk)"
          actions={
            <button
              type="button"
              onClick={() => setShowBlocked((v) => !v)}
              className="focus-ring rounded-lg border border-white/10 px-2.5 py-1 text-xs text-slate-300 hover:bg-white/5"
            >
              {showBlocked ? 'Hide blocked' : 'Show blocked'}
            </button>
          }
        >
          {recommendations.isLoading ? (
            <LoadingPanel rows={6} />
          ) : recommendations.isError ? (
            <ErrorState error={recommendations.error} onRetry={() => recommendations.refetch()} />
          ) : all.length === 0 ? (
            <EmptyState
              title="No opportunities found"
              detail="Findings worth under $10/month are suppressed. As usage grows, new opportunities will appear here."
            />
          ) : (
            <ul className="divide-y divide-white/5">
              {all.map((rec) => (
                <RecommendationRow
                  key={rec.id}
                  rec={rec}
                  expanded={expanded === rec.id}
                  onToggle={() => setExpanded(expanded === rec.id ? null : rec.id)}
                />
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Savings waterfall" subtitle="Cumulative effect on monthly spend">
          {summary.isLoading || recommendations.isLoading ? (
            <LoadingPanel rows={5} />
          ) : waterfallItems.length === 0 ? (
            <EmptyState title="Nothing to model yet" />
          ) : (
            <>
              <SavingsWaterfall baseline={currentSpend} items={waterfallItems} />
              <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                Levers are applied cumulatively, not summed independently — overlapping savings are
                netted so the final figure does not double-count the same tokens.
              </p>
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}

function RecommendationRow({
  rec,
  expanded,
  onToggle,
}: {
  rec: Recommendation;
  expanded: boolean;
  onToggle: () => void;
}) {
  const risk = RISK_STYLES[rec.risk] ?? RISK_STYLES.low;

  return (
    <li className={rec.blocked_by_quality ? 'opacity-70' : undefined}>
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        className="focus-ring flex w-full items-start justify-between gap-4 py-3 text-left"
      >
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-slate-100">{rec.title}</span>
            {rec.blocked_by_quality && <Badge tone="danger">Blocked on quality</Badge>}
          </div>
          <p className="mt-0.5 line-clamp-2 text-xs leading-relaxed text-slate-400">
            {rec.rationale}
          </p>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <Badge tone={rec.effort === 'config_change' ? 'success' : 'neutral'}>
              {rec.effort.replace(/_/g, ' ')}
            </Badge>
            <span className={`text-[11px] ${risk.text}`}>{risk.label}</span>
            {rec.requires_evaluation && <Badge tone="warning">Needs evaluation</Badge>}
            <span className="text-[11px] text-slate-500">
              ~{toNumber(rec.implementation_hours)}h
            </span>
          </div>
        </div>
        <div className="shrink-0 text-right">
          <p className="tabular text-sm font-semibold text-emerald-300">
            {formatCurrency(rec.estimated_monthly_savings, { compact: true })}
          </p>
          <p className="text-[11px] text-slate-500">
            {formatCurrency(rec.annual_savings, { compact: true })}/yr
          </p>
        </div>
      </button>

      {expanded && (
        <div className="pb-4 pl-1 pr-1">
          {rec.blocked_by_quality && rec.quality_note && (
            <div className="mb-3 rounded-lg border border-red-500/20 bg-red-500/5 p-3">
              <p className="text-xs font-medium text-red-200">Why this is blocked</p>
              <p className="mt-1 text-xs leading-relaxed text-red-300/80">{rec.quality_note}</p>
            </div>
          )}
          <p className="text-xs font-medium text-slate-300">Implementation</p>
          <ol className="mt-1.5 space-y-1.5">
            {rec.implementation_steps.map((step, i) => (
              <li key={i} className="flex gap-2 text-xs leading-relaxed text-slate-400">
                <span className="tabular shrink-0 text-slate-600">{i + 1}.</span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
          <div className="mt-3 flex flex-wrap gap-4 text-[11px] text-slate-500">
            <span>Confidence {formatPercent(toNumber(rec.confidence) * 100, 0)}</span>
            <span>Scope {rec.scope_key}</span>
            {toNumber(rec.quality_impact) !== 0 && (
              <span>Quality impact {toNumber(rec.quality_impact).toFixed(3)}</span>
            )}
          </div>
        </div>
      )}
    </li>
  );
}
