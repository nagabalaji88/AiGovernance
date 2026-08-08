# API Reference

Base path `/api/v1`. Interactive schema at `/docs` (non-production only).

## Conventions

**Money is a string.** Every monetary field serialises as a decimal string,
never a JSON number. `1234.5678901234` parsed as an IEEE-754 double loses
precision above 2^53 and reintroduces exactly the drift the backend avoids.
Clients should parse at the display boundary and no earlier.

**Timestamps are RFC 3339, UTC.** No local times anywhere. A cost report whose
day boundary depends on the reader's timezone cannot be reconciled.

**Errors are structured.**

```json
{
  "code": "budget_exceeded",
  "detail": "Budget department:eng at 104.2% of 45000 for the monthly period.",
  "request_id": "8f3a2b1c9d4e5f60",
  "fields": null
}
```

`code` is stable and machine-readable; `detail` is human prose and may be
reworded without a version bump. Never branch on `detail`. `request_id` is
echoed in `X-Request-ID` and appears in every server log line for the request.

**Pagination is cursor-based** on usage endpoints. Offset pagination is not
offered: `OFFSET 5000000` forces Postgres to walk and discard five million
rows, and the result is unstable under concurrent writes — a client walking the
list both re-sees and misses records.

## Authentication

```http
Authorization: Bearer <access_token>     # humans, 30 min TTL
X-API-Key: aick_<key>                    # SDKs, scoped, revocable
```

## Ingestion

### `POST /ingest/events` → `202 Accepted`

Accepts up to 1000 events per batch.

```json
{
  "events": [{
    "provider": "openai",
    "model": "gpt-4.1",
    "occurred_at": "2026-08-04T09:15:22Z",
    "idempotency_key": "req-8f3a2b1c",
    "tokens": {
      "input": 8420, "output": 512, "cached_input": 4096, "reasoning": 0
    },
    "attribution": {
      "team_id": "22222222-0000-0000-0000-000000000002",
      "feature": "code-review",
      "cost_center": "CC-2200",
      "environment": "production"
    },
    "trace": {
      "latency_ms": 1840,
      "conversation_id": "conv-1029",
      "conversation_turn": 4,
      "system_prompt_tokens": 3200,
      "rag_tokens": 4100,
      "rag_chunks": 10
    },
    "prompt_fingerprint": "a3f8c2e1d9b04756"
  }]
}
```

Response:

```json
{ "accepted": 1, "duplicates": 0, "rejected": 0, "unpriced": 0, "errors": [] }
```

Notes that matter in practice:

- **`202`, not `201`.** Events are queued and priced asynchronously.
  Instrumentation must never couple the caller's latency to our write path.
- **`idempotency_key` is strongly recommended.** SDKs retry; a double-counted
  charge is worse than a missing one, because finance can explain a gap but not
  a phantom.
- **`unpriced > 0` is not a client error.** The event was stored; no rate card
  matched. It raises an operational alert on our side. Do not retry.
- **`prompt_fingerprint` is a hash, never text.** It enables duplicate-request
  and cache analysis without the platform ingesting prompt content.
- **Tags are capped at 20 keys.** Unbounded tag cardinality creates a time
  series per request and takes down the metrics backend.

## Governance

### `POST /governance/preflight` → `200`

Called *before* dispatching to the provider. p99 budget: 5ms.

```json
{ "provider": "anthropic", "model": "claude-opus-5",
  "estimated_input_tokens": 400000, "estimated_output_tokens": 60000,
  "team_id": "2222...", "feature": "contract-analysis" }
```

```json
{ "action": "require_approval", "allowed": false,
  "estimated_cost": "3.5000000000",
  "reasons": ["Estimated cost 3.5000000000 requires approval (threshold 3)."],
  "violated_policies": ["Approval required for high-cost reasoning calls"],
  "suggested_model": null, "evaluated_in_ms": 0.42 }
```

Actions: `allow`, `warn`, `require_approval`, `downgrade_model`, `throttle`,
`block`. On `downgrade_model`, retry against `suggested_model`.

**Fails open.** If the service is unreachable, treat as `allow` — a governance
outage must not become an outage in your product.

### `GET /governance/budgets` · `POST /governance/budgets` · `GET /governance/chargeback?days=30`

Chargeback lines carry `direct_cost` and `allocated_shared_cost` separately.
Unattributable spend is allocated proportionally to direct usage — the method
finance already applies to shared infrastructure. Presenting only the blended
total invites a dispute nobody can settle.

## Analytics

| Endpoint | Returns |
|---|---|
| `GET /analytics/summary?days=30` | Headline cost, waste ratio, attribution coverage, period-over-period |
| `GET /analytics/breakdown?dimension=&days=` | Ranked spend by dimension |
| `GET /analytics/timeseries?days=30` | Dense daily series (gap-filled with zeros) |
| `GET /analytics/tokens?days=30` | Prompt percentiles, region composition, context growth slope |
| `GET /analytics/heatmap?row_dimension=team&column_dimension=model` | Two-dimensional cost matrix |
| `GET /analytics/flow?days=30` | Sankey links: department → team → provider → model |

Supported `dimension` values: `provider`, `model`, `model_type`, `department`,
`team`, `user`, `feature`, `project`, `application`, `environment`, `customer`,
`cost_center`, `prompt_template`, `status`.

The timeseries is **dense** — every day in range is present, zero-filled.
A sparse series makes a chart misrepresent a gap as a straight line and breaks
the forecaster's uniform-spacing assumption.

## Forecasting

### `GET /forecast?horizon_days=30&history_days=90`

```json
{
  "points": [{ "at": "2026-08-05", "value": "38.2654",
               "lower": "25.4733", "upper": "51.0575" }],
  "method": "holt_winters", "mape": "35.85", "confidence": "0.80",
  "seasonal": true, "warnings": [],
  "horizon_totals": { "7d": "268.11", "30d": "1174.32", "90d": "3702.88" }
}
```

- **`mape` is the backtested error.** Always display it. An unqualified
  forecast invites false precision.
- **`mape: null`** means history was too short to backtest — treat the forecast
  as directional only.
- **The current day is excluded** from fitting. It is always partial, which
  biases the level downward and inflates apparent error.
- **`warnings`** carries model degradation notices (insufficient history,
  seasonality not modelled). Surface them next to the chart.

## Optimization

| Endpoint | Purpose |
|---|---|
| `GET /optimize/recommendations?days=30&include_blocked=false` | All findings, ranked by priority score |
| `POST /optimize/prompt` | Analyse a prompt for removable tokens |
| `POST /optimize/route` | Recommend a model under constraints |
| `POST /optimize/rag` | Price RAG parameter changes |
| `GET /optimize/models/compare?models=openai/gpt-5,anthropic/claude-sonnet-5` | Side-by-side projection |

Recommendations are ranked by

```
priority = annual_savings × confidence ÷ (effort_weight × risk_weight)
```

which surfaces the change returning most per unit of engineering pain — not the
largest absolute number, which is invariably the hardest and riskiest.

`include_blocked=true` returns recommendations the quality gate refused,
carrying `blocked_by_quality: true` and a `quality_note` explaining why. They
are shown rather than hidden so teams do not implement the change themselves
without the gate.

`POST /optimize/prompt` returns **no prompt content** — only counts, character
offsets and a fingerprint. The offsets let a client highlight spans in the
user's own editor without the text leaving their browser.

## Simulation

### `POST /simulate`

```json
{
  "profile": { "provider": "openai", "model": "gpt-4.1",
               "monthly_requests": 500000, "avg_input_tokens": 10000,
               "avg_output_tokens": 800, "static_input_tokens": 4000,
               "rag_input_tokens": 3000 },
  "levers": [{ "type": "enable_prompt_cache", "hit_rate": "0.85" },
             { "type": "reduce_rag_context", "reduction_pct": "40" }]
}
```

Levers: `switch_model`, `compress_prompt`, `enable_prompt_cache`,
`enable_response_cache`, `reduce_rag_context`, `summarise_history`,
`batch_requests`. Omit `levers` to receive the standard scenario set.

Levers **compose on the token pipeline** — they are not summed. 30% compression
plus a 50% cheaper model is a 65% saving, not 80%.

Every scenario carries `quality_delta`, `warnings` and a `recommendation`
verdict. A scenario saving 60% while degrading answers returns
`"reject: quality degradation outweighs any saving"`. Do not surface the
savings figure without the verdict.

## Anomalies

### `GET /anomalies?days=7&min_severity=medium`

Findings are ranked by `estimated_impact` in dollars, not by statistical
score — the responder's first item should be the most expensive one, not the
most surprising one.

Kinds: `cost_spike`, `token_spike`, `latency_spike`, `error_burst`,
`retry_storm`, `runaway_agent`, `infinite_loop`, `context_explosion`,
`provider_drift`, `prompt_injection`, `api_abuse`, `rag_misconfiguration`.

Findings are rolled up per feature, not per prompt: one FAQ endpoint with
twelve repeated questions is one finding with one fix, not twelve rows.

## Catalog

### `GET /catalog/models`

The active pricing catalog — every model, its per-million-token rates by token
class, context window, quality index and capability flags. This is the same
data all cost resolution uses, so a discrepancy between a client's estimate and
a billed figure can be diagnosed from it directly.

## Rate limits

| Scope | Limit |
|---|---|
| Dashboard / query endpoints | 600 req/min per user |
| Ingestion | 60,000 req/min per organization |
| Pre-flight | Not limited — throttling the governance path would defeat its purpose |

`429` responses carry `Retry-After`.
