/**
 * What-if simulator.
 *
 * The pre-commitment surface: price a change before making it. The design
 * point is that scenarios are presented with their *warnings and quality
 * impact attached*, never as a bare savings number — a scenario that saves 60%
 * by degrading answers is a failure, and the UI should make that impossible to
 * miss rather than burying it below the fold.
 */

import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';

import { Badge, EmptyState, ErrorState, LoadingPanel, Panel } from '../components/ui';
import { api, formatCurrency, formatPercent, toNumber, type Scenario } from '../lib/api';

const DEFAULT_PROFILE = {
  provider: 'openai',
  model: 'gpt-4.1',
  monthly_requests: 500_000,
  avg_input_tokens: 10_000,
  avg_output_tokens: 800,
  static_input_tokens: 4_000,
  rag_input_tokens: 3_000,
  cacheable_request_ratio: '0.25',
};

export default function Simulator() {
  const [profile, setProfile] = useState(DEFAULT_PROFILE);
  const catalog = useQuery({ queryKey: ['catalog'], queryFn: () => api.catalog() });

  const simulate = useMutation({
    mutationFn: () => api.simulate(profile, []),
  });

  const scenarios = simulate.data ?? [];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold text-slate-50">What-if simulator</h1>
        <p className="mt-1 text-sm text-slate-400">
          Price a change before you make it. Levers compose against the token pipeline, so
          overlapping savings are netted rather than summed.
        </p>
      </header>

      <div className="grid gap-6 xl:grid-cols-3">
        <Panel title="Workload profile" subtitle="Measured from your real usage, or edited here">
          <div className="space-y-3">
            <Field label="Model">
              <select
                value={`${profile.provider}/${profile.model}`}
                onChange={(e) => {
                  const [provider, ...rest] = e.target.value.split('/');
                  setProfile({ ...profile, provider, model: rest.join('/') });
                }}
                className="focus-ring w-full rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-xs text-slate-200 [&>option]:bg-slate-900"
              >
                {(catalog.data ?? []).map((m) => (
                  <option key={`${m.provider}/${m.model}`} value={`${m.provider}/${m.model}`}>
                    {m.provider} / {m.model}
                  </option>
                ))}
              </select>
            </Field>

            <NumberField
              label="Monthly requests"
              value={profile.monthly_requests}
              onChange={(v) => setProfile({ ...profile, monthly_requests: v })}
            />
            <NumberField
              label="Avg input tokens"
              value={profile.avg_input_tokens}
              onChange={(v) => setProfile({ ...profile, avg_input_tokens: v })}
            />
            <NumberField
              label="Avg output tokens"
              value={profile.avg_output_tokens}
              onChange={(v) => setProfile({ ...profile, avg_output_tokens: v })}
            />
            <NumberField
              label="Static prefix tokens"
              hint="System prompt, tool definitions and few-shot — the part a prompt cache can cover."
              value={profile.static_input_tokens}
              onChange={(v) => setProfile({ ...profile, static_input_tokens: v })}
            />
            <NumberField
              label="RAG context tokens"
              value={profile.rag_input_tokens}
              onChange={(v) => setProfile({ ...profile, rag_input_tokens: v })}
            />

            <button
              type="button"
              onClick={() => simulate.mutate()}
              disabled={simulate.isPending}
              className="focus-ring w-full rounded-lg bg-indigo-500/90 px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-indigo-500 disabled:opacity-50"
            >
              {simulate.isPending ? 'Running…' : 'Run scenarios'}
            </button>
          </div>
        </Panel>

        <div className="space-y-6 xl:col-span-2">
          {simulate.isError && <ErrorState error={simulate.error} onRetry={() => simulate.mutate()} />}
          {simulate.isPending && (
            <Panel title="Scenarios">
              <LoadingPanel rows={5} />
            </Panel>
          )}
          {!simulate.isPending && scenarios.length === 0 && (
            <Panel title="Scenarios">
              <EmptyState
                title="Run a simulation to compare strategies"
                detail="The standard set spans the risk spectrum: safe config-only wins through to an aggressive everything-at-once scenario."
              />
            </Panel>
          )}
          {scenarios.map((scenario) => (
            <ScenarioCard key={scenario.name} scenario={scenario} />
          ))}
        </div>
      </div>
    </div>
  );
}

function ScenarioCard({ scenario }: { scenario: Scenario }) {
  const verdict = scenario.recommendation;
  const rejected = verdict.startsWith('reject');
  const conditional = verdict.startsWith('conditional');
  const savings = toNumber(scenario.monthly_savings);
  const qualityDelta = toNumber(scenario.quality_delta);

  return (
    <Panel
      title={scenario.name}
      subtitle={scenario.levers.join(' · ')}
      className={rejected ? 'opacity-75' : undefined}
      actions={
        <Badge tone={rejected ? 'danger' : conditional ? 'warning' : 'success'}>
          {verdict.split(':')[0]}
        </Badge>
      }
    >
      <div className="grid gap-4 sm:grid-cols-4">
        <Metric label="Monthly saving" value={formatCurrency(savings, { compact: true })} highlight={!rejected} />
        <Metric label="Annualised" value={formatCurrency(scenario.annual_savings, { compact: true })} />
        <Metric label="Reduction" value={formatPercent(scenario.savings_pct, 0)} />
        <Metric
          label="Quality impact"
          value={qualityDelta === 0 ? 'None' : qualityDelta.toFixed(3)}
          tone={qualityDelta < -0.02 ? 'danger' : qualityDelta < 0 ? 'warning' : 'neutral'}
        />
      </div>

      <p className="mt-4 text-xs leading-relaxed text-slate-300">{verdict}</p>

      {scenario.warnings.length > 0 && (
        <ul className="mt-3 space-y-1.5 rounded-lg border border-amber-500/20 bg-amber-500/5 p-3">
          {scenario.warnings.map((warning) => (
            <li key={warning} className="text-xs leading-relaxed text-amber-200/90">
              ⚠ {warning}
            </li>
          ))}
        </ul>
      )}

      {scenario.notes.length > 0 && (
        <details className="mt-3">
          <summary className="focus-ring cursor-pointer text-xs text-slate-400 hover:text-slate-300">
            How this was calculated
          </summary>
          <ul className="mt-2 space-y-1">
            {scenario.notes.map((note) => (
              <li key={note} className="text-[11px] leading-relaxed text-slate-500">
                • {note}
              </li>
            ))}
          </ul>
        </details>
      )}
    </Panel>
  );
}

function Metric({
  label,
  value,
  highlight,
  tone = 'neutral',
}: {
  label: string;
  value: string;
  highlight?: boolean;
  tone?: 'neutral' | 'warning' | 'danger';
}) {
  const color =
    tone === 'danger'
      ? 'text-red-300'
      : tone === 'warning'
        ? 'text-amber-300'
        : highlight
          ? 'text-emerald-300'
          : 'text-slate-100';
  return (
    <div>
      <p className="text-[11px] uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`tabular mt-1 text-lg font-semibold ${color}`}>{value}</p>
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="text-[11px] font-medium text-slate-400">{label}</span>
      <div className="mt-1">{children}</div>
      {hint && <span className="mt-1 block text-[10px] leading-relaxed text-slate-600">{hint}</span>}
    </label>
  );
}

function NumberField({
  label,
  value,
  onChange,
  hint,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  hint?: string;
}) {
  return (
    <Field label={label} hint={hint}>
      <input
        type="number"
        min={0}
        value={value}
        onChange={(e) => onChange(Math.max(0, Number(e.target.value) || 0))}
        className="tabular focus-ring w-full rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-xs text-slate-200"
      />
    </Field>
  );
}
