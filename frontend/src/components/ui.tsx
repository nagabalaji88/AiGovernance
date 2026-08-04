/**
 * Shared presentational primitives.
 *
 * Every data-bearing component here handles three states explicitly: loading,
 * error, and empty. That is deliberate — a dashboard that renders a blank
 * panel on failure teaches users to distrust every panel, including the ones
 * that are working. An empty state that says *why* it is empty ("no spend in
 * this window") is information; a blank rectangle is a bug report.
 */

import { motion } from 'framer-motion';
import type { ReactNode } from 'react';
import clsx from 'clsx';

import { SEVERITY_STYLES } from '../lib/theme';

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
  padded = true,
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  padded?: boolean;
}) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
      className={clsx('glass overflow-hidden', className)}
    >
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-white/5 px-5 py-4">
          <div>
            {title && <h2 className="text-sm font-semibold text-slate-100">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={padded ? 'p-5' : undefined}>{children}</div>
    </motion.section>
  );
}

export function StatTile({
  label,
  value,
  delta,
  deltaLabel,
  hint,
  tone = 'neutral',
  index = 0,
}: {
  label: string;
  value: string;
  delta?: number | null;
  deltaLabel?: string;
  hint?: string;
  tone?: 'neutral' | 'positive' | 'negative';
  index?: number;
}) {
  // For cost metrics a rise is bad, so the arrow colour is driven by an
  // explicit `tone` rather than by the sign of the delta. Auto-colouring by
  // sign is how dashboards end up painting a 40% cost increase green.
  const deltaTone =
    delta === null || delta === undefined
      ? 'text-slate-400'
      : tone === 'positive'
        ? delta >= 0
          ? 'text-emerald-300'
          : 'text-red-300'
        : delta >= 0
          ? 'text-red-300'
          : 'text-emerald-300';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.04, ease: [0.22, 1, 0.36, 1] }}
      className="glass glass-hover p-5"
    >
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <p className="tabular mt-2 text-2xl font-semibold text-slate-50">{value}</p>
      <div className="mt-1.5 flex items-center gap-2 text-xs">
        {delta !== null && delta !== undefined && (
          <span className={clsx('tabular font-medium', deltaTone)}>
            {delta >= 0 ? '▲' : '▼'} {Math.abs(delta).toFixed(1)}%
          </span>
        )}
        {deltaLabel && <span className="text-slate-500">{deltaLabel}</span>}
      </div>
      {hint && <p className="mt-2 text-[11px] leading-relaxed text-slate-500">{hint}</p>}
    </motion.div>
  );
}

/**
 * Severity chip. Colour *and* text, never colour alone — roughly 8% of men
 * have a colour vision deficiency, and severity is exactly the signal that
 * must survive it.
 */
export function SeverityChip({ severity }: { severity: string }) {
  const style = SEVERITY_STYLES[severity] ?? SEVERITY_STYLES.info;
  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ring-1',
        style.bg,
        style.text,
        style.ring,
      )}
    >
      {style.label}
    </span>
  );
}

export function Badge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode;
  tone?: 'neutral' | 'accent' | 'success' | 'warning' | 'danger';
}) {
  const tones = {
    neutral: 'bg-slate-500/12 text-slate-300 ring-slate-400/20',
    accent: 'bg-indigo-500/15 text-indigo-300 ring-indigo-400/25',
    success: 'bg-emerald-500/12 text-emerald-300 ring-emerald-400/25',
    warning: 'bg-amber-500/12 text-amber-300 ring-amber-400/25',
    danger: 'bg-red-500/12 text-red-300 ring-red-400/25',
  };
  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-medium ring-1',
        tones[tone],
      )}
    >
      {children}
    </span>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx('animate-pulse rounded-lg bg-white/5', className)} />;
}

export function LoadingPanel({ rows = 4 }: { rows?: number }) {
  return (
    <div className="space-y-3" aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading</span>
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-10 w-full" />
      ))}
    </div>
  );
}

/**
 * Error state that stays actionable.
 *
 * Shows the request id when the server supplied one — that single string is
 * the difference between "the dashboard is broken" and a support ticket an
 * engineer can resolve by grepping the logs.
 */
export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof Error ? error.message : 'Something went wrong.';
  const requestId =
    error && typeof error === 'object' && 'requestId' in error
      ? (error as { requestId?: string }).requestId
      : undefined;

  return (
    <div role="alert" className="rounded-xl border border-red-500/20 bg-red-500/5 p-5 text-sm">
      <p className="font-medium text-red-200">Could not load this panel</p>
      <p className="mt-1 text-red-300/80">{message}</p>
      {requestId && (
        <p className="tabular mt-2 text-xs text-red-300/60">Request ID: {requestId}</p>
      )}
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="focus-ring mt-3 rounded-lg border border-red-400/30 px-3 py-1.5 text-xs font-medium text-red-200 hover:bg-red-500/10"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-white/10 px-6 py-10 text-center">
      <p className="text-sm font-medium text-slate-300">{title}</p>
      {detail && <p className="mt-1 max-w-sm text-xs leading-relaxed text-slate-500">{detail}</p>}
    </div>
  );
}

export function ProgressBar({
  value,
  max = 100,
  color,
  label,
}: {
  value: number;
  max?: number;
  color: string;
  label?: string;
}) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
      className="h-2 w-full overflow-hidden rounded-full bg-white/8"
    >
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="h-full rounded-full"
        style={{ backgroundColor: color }}
      />
    </div>
  );
}

export function Select({
  value,
  onChange,
  options,
  label,
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  label: string;
}) {
  return (
    <label className="flex items-center gap-2 text-xs text-slate-400">
      <span className="sr-only sm:not-sr-only">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="focus-ring rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-xs text-slate-200 [&>option]:bg-slate-900"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
