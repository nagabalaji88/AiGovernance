/**
 * Chart components.
 *
 * All charts share a few non-negotiable rules, applied here once rather than
 * per-page:
 *
 * - **Y axes start at zero for cost.** A truncated axis exaggerates a 3%
 *   variation into a visual cliff. On a page where people make budget
 *   decisions, that is not a styling choice, it is a misrepresentation.
 * - **Forecast bands are drawn, not implied.** A single forecast line reads as
 *   certainty. The confidence interval is rendered as an area behind it so the
 *   uncertainty is impossible to miss.
 * - **Series colour comes from the categorical ramp only**, never the semantic
 *   red/amber/green, which is reserved for budget health.
 */

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { formatCurrency, formatNumber, toNumber } from '../lib/api';
import { CATEGORICAL, SEMANTIC, seriesColor } from '../lib/theme';

const AXIS = { stroke: 'rgba(148,163,184,0.25)', fontSize: 11 };

function shortDate(value: string): string {
  const d = new Date(value);
  return Number.isNaN(d.getTime())
    ? value
    : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export function CostTrend({
  data,
  height = 260,
}: {
  data: { at: string; cost: string; tokens: number }[];
  height?: number;
}) {
  const series = data.map((d) => ({ at: d.at, cost: toNumber(d.cost), tokens: d.tokens }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={series} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="costFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={CATEGORICAL[0]} stopOpacity={0.42} />
            <stop offset="100%" stopColor={CATEGORICAL[0]} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="at" tickFormatter={shortDate} {...AXIS} tickLine={false} />
        {/* domain starts at 0: truncated cost axes exaggerate small moves. */}
        <YAxis
          domain={[0, 'auto']}
          tickFormatter={(v) => formatCurrency(v, { compact: true })}
          {...AXIS}
          tickLine={false}
          axisLine={false}
          width={62}
        />
        <Tooltip
          formatter={(value: number) => [formatCurrency(value), 'Cost']}
          labelFormatter={shortDate}
        />
        <Area
          type="monotone"
          dataKey="cost"
          stroke={CATEGORICAL[0]}
          strokeWidth={2}
          fill="url(#costFill)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function ForecastChart({
  history,
  forecast,
  height = 300,
}: {
  history: { at: string; cost: string }[];
  forecast: { at: string; value: string; lower: string; upper: string }[];
  height?: number;
}) {
  // The band is drawn as a stacked pair: a transparent floor at `lower` plus a
  // visible span of (upper - lower). Recharts has no native band mark, and
  // this is the standard composition for one.
  const rows = [
    ...history.map((h) => ({
      at: h.at,
      actual: toNumber(h.cost),
      forecast: null as number | null,
      lower: null as number | null,
      span: null as number | null,
    })),
    ...forecast.map((f) => ({
      at: f.at,
      actual: null as number | null,
      forecast: toNumber(f.value),
      lower: toNumber(f.lower),
      span: toNumber(f.upper) - toNumber(f.lower),
    })),
  ];

  return (
    <ResponsiveContainer width="100%" height={height}>
      {/* ComposedChart, not AreaChart: mixing Line marks into an AreaChart
          silently drops them, which left the actual-cost line invisible and
          made the panel look like it had no history at all. */}
      <ComposedChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="at" tickFormatter={shortDate} {...AXIS} tickLine={false} minTickGap={28} />
        <YAxis
          domain={[0, 'auto']}
          tickFormatter={(v) => formatCurrency(v, { compact: true })}
          {...AXIS}
          tickLine={false}
          axisLine={false}
          width={62}
        />
        <Tooltip
          formatter={(value: number, name: string) => {
            if (name === 'span' || name === 'lower') return [null, null];
            return [formatCurrency(value), name === 'actual' ? 'Actual' : 'Forecast'];
          }}
          labelFormatter={shortDate}
        />
        <Area
          dataKey="lower"
          stackId="band"
          stroke="none"
          fill="transparent"
          isAnimationActive={false}
          legendType="none"
        />
        <Area
          dataKey="span"
          stackId="band"
          stroke="none"
          fill={CATEGORICAL[2]}
          fillOpacity={0.16}
          isAnimationActive={false}
          legendType="none"
        />
        <Line type="monotone" dataKey="actual" stroke={CATEGORICAL[0]} strokeWidth={2} dot={false} />
        <Line
          type="monotone"
          dataKey="forecast"
          stroke={CATEGORICAL[2]}
          strokeWidth={2}
          strokeDasharray="5 4"
          dot={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function BreakdownBars({
  data,
  height = 280,
}: {
  data: { key: string; cost: string }[];
  height?: number;
}) {
  const rows = data.slice(0, 10).map((d) => ({ key: d.key, cost: toNumber(d.cost) }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis
          type="number"
          tickFormatter={(v) => formatCurrency(v, { compact: true })}
          {...AXIS}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          type="category"
          dataKey="key"
          width={150}
          {...AXIS}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip formatter={(value: number) => [formatCurrency(value), 'Cost']} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
        <Bar dataKey="cost" radius={[0, 6, 6, 0]} maxBarSize={22}>
          {rows.map((_, i) => (
            <Cell key={i} fill={seriesColor(i)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function CompositionDonut({
  composition,
  height = 260,
}: {
  composition: Record<string, number>;
  height?: number;
}) {
  const rows = Object.entries(composition)
    .filter(([, v]) => v > 0.5)
    .map(([name, value]) => ({
      name: name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
      value,
    }))
    .sort((a, b) => b.value - a.value);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={rows}
          dataKey="value"
          nameKey="name"
          innerRadius="55%"
          outerRadius="80%"
          paddingAngle={2}
          stroke="none"
        >
          {rows.map((_, i) => (
            <Cell key={i} fill={seriesColor(i)} />
          ))}
        </Pie>
        <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, 'Share of prompt']} />
        <Legend
          verticalAlign="bottom"
          height={48}
          iconType="circle"
          formatter={(value) => <span style={{ color: '#94a3b8', fontSize: 11 }}>{value}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}

/**
 * Cost heatmap.
 *
 * Uses a single-hue sequential ramp (opacity over one accent), not a
 * red-to-green diverging scale. Diverging scales imply a meaningful midpoint,
 * and there is none here — $500 of spend is not "neutral". A sequential ramp
 * says "more" without implying "worse".
 */
export function CostHeatmap({
  rows,
  columns,
  values,
}: {
  rows: string[];
  columns: string[];
  values: number[][];
}) {
  const max = Math.max(...values.flat(), 1);

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-1 text-xs">
        <thead>
          <tr>
            <th className="sticky left-0 z-10 bg-transparent p-1 text-left font-medium text-slate-400" />
            {columns.map((c) => (
              <th key={c} className="p-1 text-left font-medium text-slate-400">
                <span className="block max-w-[7rem] truncate" title={c}>
                  {c}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={row}>
              <th
                scope="row"
                className="sticky left-0 z-10 max-w-[9rem] truncate p-1 text-left font-medium text-slate-300"
                title={row}
              >
                {row}
              </th>
              {columns.map((col, ci) => {
                const value = values[ri]?.[ci] ?? 0;
                const intensity = value / max;
                return (
                  <td key={col} className="p-0">
                    <div
                      className="tabular flex h-9 min-w-[5rem] items-center justify-center rounded-md text-[11px] text-slate-100"
                      style={{
                        // Floor at 0.04 so a populated-but-tiny cell is still
                        // visually distinct from a genuinely empty one.
                        backgroundColor:
                          value > 0
                            ? `rgba(99, 102, 241, ${Math.max(0.08, intensity * 0.85)})`
                            : 'rgba(148,163,184,0.04)',
                      }}
                      title={`${row} × ${col}: ${formatCurrency(value)}`}
                    >
                      {value > 0 ? formatCurrency(value, { compact: true }) : '—'}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Cost flow, rendered as a ranked flow list rather than a true Sankey.
 *
 * A real Sankey needs a layout solver and, at the node counts an enterprise
 * tenant produces (dozens of teams x dozens of models), degenerates into
 * unreadable spaghetti. A ranked list of weighted flows conveys the same
 * information — where the money goes, in order — and stays legible. The
 * `/analytics/flow` endpoint returns proper Sankey triples, so a graph
 * renderer can be swapped in for tenants with small enough topologies.
 */
export function CostFlow({ links }: { links: { source: string; target: string; value: number }[] }) {
  const top = links.slice(0, 12);
  const max = Math.max(...top.map((l) => l.value), 1);

  return (
    <ul className="space-y-2">
      {top.map((link, i) => (
        <li key={`${link.source}->${link.target}`} className="group">
          <div className="flex items-center justify-between gap-3 text-xs">
            <span className="truncate text-slate-300">
              <span className="text-slate-400">{link.source.split(':').pop()}</span>
              <span className="mx-1.5 text-slate-600">→</span>
              <span className="font-medium">{link.target.split(':').pop()}</span>
            </span>
            <span className="tabular shrink-0 text-slate-400">
              {formatCurrency(link.value, { compact: true })}
            </span>
          </div>
          <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-white/5">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${(link.value / max) * 100}%`,
                backgroundColor: seriesColor(i),
                opacity: 0.75,
              }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

export function SavingsWaterfall({
  baseline,
  items,
  height = 260,
}: {
  baseline: number;
  items: { label: string; amount: number }[];
  height?: number;
}) {
  // Cumulative running total so each bar starts where the previous ended —
  // the defining property of a waterfall, and the thing that makes the
  // contribution of each lever readable at a glance.
  let running = baseline;
  const rows = [
    { label: 'Current', base: 0, value: baseline, isTotal: true },
    ...items.map((item) => {
      const base = running - item.amount;
      running = base;
      return { label: item.label, base, value: item.amount, isTotal: false };
    }),
    { label: 'Optimised', base: 0, value: Math.max(running, 0), isTotal: true },
  ];

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="label" {...AXIS} tickLine={false} interval={0} angle={-18} textAnchor="end" height={58} />
        <YAxis
          tickFormatter={(v) => formatCurrency(v, { compact: true })}
          {...AXIS}
          tickLine={false}
          axisLine={false}
          width={62}
        />
        <Tooltip
          formatter={(value: number, name) =>
            name === 'base' ? [null, null] : [formatCurrency(value), 'Monthly']
          }
          cursor={{ fill: 'rgba(255,255,255,0.04)' }}
        />
        <Bar dataKey="base" stackId="w" fill="transparent" isAnimationActive={false} />
        <Bar dataKey="value" stackId="w" radius={[6, 6, 0, 0]} maxBarSize={54}>
          {rows.map((row, i) => (
            <Cell key={i} fill={row.isTotal ? CATEGORICAL[0] : SEMANTIC.savings} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function QualityCostScatter({
  models,
  height = 300,
}: {
  models: { model: string; provider: string; cost: number; quality: number }[];
  height?: number;
}) {
  // Rendered as an annotated grid rather than a Recharts scatter so each point
  // can carry its label — with 30 models, an unlabelled scatter forces the
  // user to hover every dot to answer "which one is that".
  const maxCost = Math.max(...models.map((m) => m.cost), 0.000001);

  return (
    <div className="relative" style={{ height }}>
      <div className="absolute inset-0 rounded-xl border border-white/5 bg-white/[0.02]" />
      <div className="absolute inset-x-0 bottom-0 flex justify-between px-3 pb-1 text-[10px] text-slate-500">
        <span>Cheaper →</span>
        <span>← More expensive</span>
      </div>
      <div className="absolute inset-y-0 left-0 flex flex-col justify-between py-3 pl-1 text-[10px] text-slate-500">
        <span>Higher quality</span>
        <span>Lower quality</span>
      </div>
      {models.map((m, i) => {
        // Log scale on cost: model prices span three orders of magnitude, and
        // a linear axis collapses every cheap model into the left margin.
        const x = Math.log10(1 + (m.cost / maxCost) * 999) / 3;
        const y = 1 - m.quality;
        return (
          <div
            key={`${m.provider}/${m.model}`}
            className="group absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${8 + x * 82}%`, top: `${10 + y * 74}%` }}
          >
            <div
              className="h-2.5 w-2.5 rounded-full ring-2 ring-slate-900/60"
              style={{ backgroundColor: seriesColor(i) }}
            />
            <span className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 whitespace-nowrap rounded bg-slate-900/90 px-1.5 py-0.5 text-[10px] text-slate-200 opacity-0 transition-opacity group-hover:opacity-100">
              {m.model} · {formatCurrency(m.cost)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function TokenBars({
  data,
  height = 240,
}: {
  data: { key: string; total_tokens: number }[];
  height?: number;
}) {
  const rows = data.slice(0, 8).map((d) => ({ key: d.key, tokens: d.total_tokens }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="key" {...AXIS} tickLine={false} interval={0} angle={-20} textAnchor="end" height={62} />
        <YAxis
          tickFormatter={(v) => formatNumber(v, { compact: true })}
          {...AXIS}
          tickLine={false}
          axisLine={false}
          width={54}
        />
        <Tooltip
          formatter={(value: number) => [formatNumber(value), 'Tokens']}
          cursor={{ fill: 'rgba(255,255,255,0.04)' }}
        />
        <Bar dataKey="tokens" radius={[6, 6, 0, 0]} maxBarSize={40}>
          {rows.map((_, i) => (
            <Cell key={i} fill={seriesColor(i + 3)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
