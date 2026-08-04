/**
 * Design tokens.
 *
 * ## Colour: semantic, not decorative
 *
 * In a cost product, colour carries meaning and must never be chosen for
 * looks alone. Green/amber/red are reserved exclusively for budget health and
 * severity. Categorical series (providers, models, teams) draw from a separate
 * hue-rotated ramp with no overlap into the semantic set — otherwise a team
 * whose series happens to render red reads as "over budget" at a glance, which
 * is the exact misreading a FinOps dashboard cannot afford.
 *
 * ## Accessibility over aesthetics
 *
 * Glassmorphism is the requested visual language and it fights legibility:
 * translucent surfaces mean text contrast varies with whatever sits behind it.
 * Two rules keep it usable:
 *
 * 1. Text never sits directly on a blurred surface — every glass panel has an
 *    opaque-enough base layer (`--surface`) beneath the blur to hold contrast
 *    at >= 4.5:1 for body text.
 * 2. Status is never encoded by colour alone. Every severity chip pairs its
 *    colour with a label, so the ~8% of users with colour vision deficiency
 *    read the same information.
 */

export const CATEGORICAL = [
  '#6366f1', // indigo
  '#06b6d4', // cyan
  '#8b5cf6', // violet
  '#0ea5e9', // sky
  '#a855f7', // purple
  '#14b8a6', // teal
  '#3b82f6', // blue
  '#c026d3', // fuchsia
  '#0891b2', // dark cyan
  '#7c3aed', // deep violet
] as const;

/** Reserved for budget health and anomaly severity. Never used for series. */
export const SEMANTIC = {
  healthy: '#10b981',
  caution: '#f59e0b',
  danger: '#ef4444',
  critical: '#dc2626',
  neutral: '#64748b',
  savings: '#22c55e',
} as const;

export const SEVERITY_STYLES: Record<
  string,
  { bg: string; text: string; ring: string; label: string }
> = {
  critical: { bg: 'bg-red-500/15', text: 'text-red-300', ring: 'ring-red-500/30', label: 'Critical' },
  high: { bg: 'bg-orange-500/15', text: 'text-orange-300', ring: 'ring-orange-500/30', label: 'High' },
  medium: { bg: 'bg-amber-500/15', text: 'text-amber-300', ring: 'ring-amber-500/30', label: 'Medium' },
  low: { bg: 'bg-sky-500/15', text: 'text-sky-300', ring: 'ring-sky-500/30', label: 'Low' },
  info: { bg: 'bg-slate-500/15', text: 'text-slate-300', ring: 'ring-slate-500/30', label: 'Info' },
};

export const RISK_STYLES: Record<string, { text: string; label: string }> = {
  none: { text: 'text-emerald-300', label: 'No risk' },
  low: { text: 'text-sky-300', label: 'Low risk' },
  medium: { text: 'text-amber-300', label: 'Medium risk' },
  high: { text: 'text-red-300', label: 'High risk' },
};

export function seriesColor(index: number): string {
  return CATEGORICAL[index % CATEGORICAL.length];
}

/**
 * Budget health colour.
 *
 * Thresholds match the backend's `BudgetStatus.severity` exactly. Divergence
 * between what the API considers "high" and what the UI paints amber is the
 * kind of quiet inconsistency that erodes trust in every other number on the
 * page.
 */
export function budgetColor(utilisationPct: number): string {
  if (utilisationPct >= 100) return SEMANTIC.critical;
  if (utilisationPct >= 95) return SEMANTIC.danger;
  if (utilisationPct >= 80) return SEMANTIC.caution;
  return SEMANTIC.healthy;
}
