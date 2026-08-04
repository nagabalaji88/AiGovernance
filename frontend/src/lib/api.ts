/**
 * Typed API client.
 *
 * ## Money crosses the wire as a string
 *
 * The backend serialises every monetary value as a decimal string, and this
 * client keeps it that way until the moment of display. Parsing "1234.5678901"
 * into a JS `number` loses precision above 2^53 and, more subtly, reintroduces
 * exactly the binary-float drift the backend went to some trouble to avoid.
 * `formatCurrency` parses once, at the render boundary, where a rounded display
 * value is all that is needed anyway.
 */

const BASE = import.meta.env.VITE_API_BASE_URL ?? '/api/v1';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
    readonly requestId?: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    // Surface the server's machine-readable code and request id so a user can
    // quote it in a support ticket and it can be found in the logs directly.
    let code: string | undefined;
    let detail = response.statusText;
    try {
      const body = await response.json();
      code = body.code;
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body; keep the status text */
    }
    throw new ApiError(detail, response.status, code, response.headers.get('X-Request-ID') ?? undefined);
  }
  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Response types (mirrors app/api/schemas.py)
// ---------------------------------------------------------------------------

export interface CostSummary {
  total_cost: string;
  token_cost: string;
  infrastructure_cost: string;
  cache_savings: string;
  wasted_cost: string;
  request_count: number;
  total_tokens: number;
  avg_cost_per_request: string;
  cost_per_1k_tokens: string;
  waste_ratio_pct: string;
  attribution_coverage_pct: string;
  period_over_period_pct: string | null;
}

export interface DimensionSlice {
  key: string;
  label: string | null;
  requests: number;
  total_tokens: number;
  cost: string;
  wasted_cost: string;
  cache_savings: string;
  avg_cost_per_request: string;
  avg_latency_ms: string;
  share_pct: string;
}

export interface TimeSeriesPoint {
  at: string;
  cost: string;
  tokens: number;
  requests: number;
}

export interface TokenAnalytics {
  prompt_p50: number;
  prompt_p95: number;
  prompt_p99: number;
  completion_p50: number;
  completion_p95: number;
  context_p95: number;
  composition: Record<string, number>;
  static_share_pct: string;
  context_growth_per_turn: string;
  window_utilisation_pct: string;
}

export interface ForecastPoint {
  at: string;
  value: string;
  lower: string;
  upper: string;
}

export interface Forecast {
  points: ForecastPoint[];
  method: string;
  mape: string | null;
  confidence: string;
  seasonal: boolean;
  warnings: string[];
  horizon_totals: Record<string, string>;
}

export interface Recommendation {
  id: string;
  kind: string;
  title: string;
  rationale: string;
  scope: string;
  scope_key: string;
  estimated_monthly_savings: string;
  annual_savings: string;
  confidence: string;
  priority_score: string;
  severity: string;
  effort: string;
  implementation_hours: string;
  risk: string;
  quality_impact: string;
  requires_evaluation: boolean;
  blocked_by_quality: boolean;
  quality_note: string | null;
  implementation_steps: string[];
  evidence: Record<string, unknown>;
  status: string;
}

export interface Anomaly {
  id: string | null;
  kind: string;
  severity: string;
  title: string;
  detail: string;
  scope: string;
  scope_key: string | null;
  observed_value: string;
  expected_value: string;
  deviation_score: string;
  estimated_impact: string;
  evidence: Record<string, unknown>;
  recommended_action: string | null;
  detected_at: string;
  is_resolved: boolean;
}

export interface BudgetStatus {
  id: string;
  name: string;
  scope: string;
  scope_id: string;
  amount: string;
  spent: string;
  remaining: string;
  utilisation_pct: string;
  period: string;
  period_start: string;
  period_end: string;
  severity: string;
  is_exceeded: boolean;
  projected_to_exceed: boolean;
  forecast_period_spend: string | null;
}

export interface ChargebackLine {
  cost_center: string;
  department: string | null;
  direct_cost: string;
  allocated_shared_cost: string;
  total: string;
  tokens: number;
  requests: number;
  share_pct: string;
}

export interface SankeyLink {
  source: string;
  target: string;
  value: number;
}

export interface Heatmap {
  rows: string[];
  columns: string[];
  values: number[][];
}

export interface Scenario {
  name: string;
  baseline_monthly_cost: string;
  projected_monthly_cost: string;
  monthly_savings: string;
  annual_savings: string;
  savings_pct: string;
  token_reduction_pct: string;
  quality_delta: string;
  latency_multiplier: string;
  levers: string[];
  notes: string[];
  warnings: string[];
  recommendation: string;
}

export interface CatalogModel {
  provider: string;
  model: string;
  model_type: string;
  rates_per_million: Record<string, number>;
  context_window: number;
  max_output_tokens: number;
  quality_index: number;
  latency_ms_per_1k_output: number;
  supports_prompt_cache: boolean;
  supports_batch: boolean;
  supports_vision: boolean;
  effective_from: string;
}

export interface WorkloadProfileInput {
  provider: string;
  model: string;
  monthly_requests: number;
  avg_input_tokens: number;
  avg_output_tokens: number;
  static_input_tokens?: number;
  rag_input_tokens?: number;
  cacheable_request_ratio?: string;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const api = {
  summary: (days: number) => request<CostSummary>(`/analytics/summary?days=${days}`),
  breakdown: (dimension: string, days: number, limit = 20) =>
    request<DimensionSlice[]>(
      `/analytics/breakdown?dimension=${dimension}&days=${days}&limit=${limit}`,
    ),
  timeseries: (days: number) => request<TimeSeriesPoint[]>(`/analytics/timeseries?days=${days}`),
  tokens: (days: number) => request<TokenAnalytics>(`/analytics/tokens?days=${days}`),
  heatmap: (row: string, column: string, days: number) =>
    request<Heatmap>(`/analytics/heatmap?row_dimension=${row}&column_dimension=${column}&days=${days}`),
  flow: (days: number) => request<SankeyLink[]>(`/analytics/flow?days=${days}`),
  forecast: (horizonDays: number, historyDays = 90) =>
    request<Forecast>(`/forecast?horizon_days=${horizonDays}&history_days=${historyDays}`),
  recommendations: (days: number, includeBlocked = false) =>
    request<Recommendation[]>(
      `/optimize/recommendations?days=${days}&include_blocked=${includeBlocked}`,
    ),
  anomalies: (days: number, minSeverity = 'low') =>
    request<Anomaly[]>(`/anomalies?days=${days}&min_severity=${minSeverity}`),
  budgets: () => request<BudgetStatus[]>('/governance/budgets'),
  chargeback: (days: number) => request<ChargebackLine[]>(`/governance/chargeback?days=${days}`),
  catalog: () => request<CatalogModel[]>('/catalog/models'),
  simulate: (profile: WorkloadProfileInput, levers: unknown[], name = 'scenario') =>
    request<Scenario[]>('/simulate', {
      method: 'POST',
      body: JSON.stringify({
        profile,
        levers,
        scenario_name: name,
        include_standard_scenarios: levers.length === 0,
      }),
    }),
  analysePrompt: (text: string, monthlyCalls: number) =>
    request<{
      fingerprint: string;
      total_tokens: number;
      recoverable_tokens: number;
      compression_ratio_pct: string;
      cacheable_ratio_pct: string;
      efficiency_score: string;
      monthly_savings: string;
      annual_savings: string;
      quality_risk: string;
      findings: {
        kind: string;
        detail: string;
        tokens_saved: number;
        confidence: number;
        span: number[] | null;
        severity: string;
      }[];
    }>('/optimize/prompt', {
      method: 'POST',
      body: JSON.stringify({ text, monthly_calls: monthlyCalls }),
    }),
};

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/**
 * Currency formatting that stays readable across six orders of magnitude.
 *
 * A cost dashboard shows $0.0032 per request next to $1.4M annualised. One
 * format cannot serve both: fixed 2dp turns the former into "$0.00", and
 * compact notation turns the latter into an unreadable "$1400000". The
 * threshold switch below is what keeps both legible in the same table.
 */
export function formatCurrency(value: string | number, opts?: { compact?: boolean }): string {
  const n = typeof value === 'string' ? Number.parseFloat(value) : value;
  if (!Number.isFinite(n)) return '—';
  if (n === 0) return '$0';

  const abs = Math.abs(n);
  // Compact only above 10k. Below that it destroys meaningful precision — a
  // headline of "$1K" for $1,006 of spend reads as a placeholder, and the
  // four-digit figure fits in the tile anyway.
  if (opts?.compact && abs >= 10_000) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      notation: 'compact',
      maximumFractionDigits: 1,
    }).format(n);
  }
  // Sub-cent values need significant digits, not fixed decimals, or per-request
  // costs all collapse to $0.00.
  const fractionDigits = abs < 0.01 ? 6 : abs < 1 ? 4 : 2;
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: abs < 1 ? fractionDigits : 2,
    maximumFractionDigits: fractionDigits,
  }).format(n);
}

export function formatNumber(value: number, opts?: { compact?: boolean }): string {
  if (!Number.isFinite(value)) return '—';
  return new Intl.NumberFormat('en-US', {
    notation: opts?.compact && Math.abs(value) >= 10_000 ? 'compact' : 'standard',
    maximumFractionDigits: 1,
  }).format(value);
}

export function formatPercent(value: string | number, digits = 1): string {
  const n = typeof value === 'string' ? Number.parseFloat(value) : value;
  if (!Number.isFinite(n)) return '—';
  return `${n.toFixed(digits)}%`;
}

export function formatTokens(value: number): string {
  return formatNumber(value, { compact: true });
}

export function toNumber(value: string | number | null | undefined): number {
  if (value === null || value === undefined) return 0;
  const n = typeof value === 'string' ? Number.parseFloat(value) : value;
  return Number.isFinite(n) ? n : 0;
}
