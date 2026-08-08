# Architecture

## 1. Problem

Enterprise LLM adoption produces spend that nobody can predict, attribute or
defend. The proximate causes are well known; what is missing is measurement
that connects a cost to a code change.

| Driver | Why it is invisible | What the platform does |
|---|---|---|
| Multiple providers, incompatible pricing | Each bills differently and reports usage under different field names | One normalised token vocabulary, one effective-dated catalog |
| Prompt bloat | Nobody re-reads a system prompt after it ships | Decomposes input tokens by region and prices each |
| Conversation growth | Cost grows quadratically in turn count | Measures the growth slope per conversation |
| RAG over-retrieval | `top_k` was tuned for recall with no cost term | Prices each lever and estimates the recall trade |
| Wrong model for the task | No feedback loop from task to model choice | Routes on cost/latency/quality with a quality floor |
| Redundant calls | Cache misses look identical to cache hits from outside | Fingerprint-based duplicate detection |
| Failures and retries | Billed, but produce nothing | Isolated and reported as wasted spend |
| No attribution | Keys get shared, services get renamed | Attribution stamped at emit time, never inferred later |

## 2. Requirements

### Functional

| # | Requirement | Where |
|---|---|---|
| F1 | Ingest usage from 15 providers, idempotently | `api/routes.py`, `store.ingest` |
| F2 | Price against effective-dated rate cards | `services/cost_engine.py` |
| F3 | Track nine billable token classes separately | `domain/usage.py` |
| F4 | Attribute cost to org/dept/team/user/feature/customer | `domain/usage.py` |
| F5 | Decompose prompts into cost-bearing regions | `services/token_analytics.py` |
| F6 | Forecast daily → annual with confidence intervals | `services/forecasting.py` |
| F7 | Project budget exhaustion dates | `services/forecasting.py` |
| F8 | Detect cost, token, agent, retry and cache anomalies | `services/anomaly.py` |
| F9 | Recommend prompt, model, RAG and cache optimizations | `services/recommender.py` |
| F10 | Simulate changes before execution | `services/simulator.py` |
| F11 | Enforce budgets and policies pre-flight | `services/governance.py` |
| F12 | Produce chargeback/showback statements | `services/governance.py` |
| F13 | Gate optimizations on measured quality | `services/quality.py` |
| F14 | Role-based dashboards | `frontend/src/pages/` |

### Non-functional

| # | Requirement | Target | Rationale |
|---|---|---|---|
| N1 | Dashboard query latency | p95 < 200ms | Below the threshold where a UI feels interactive |
| N2 | Pre-flight policy latency | p99 < 5ms | Above this, teams disable enforcement and the platform governs nothing |
| N3 | Ingestion throughput | 50k events/sec/cluster | A 10k-engineer org at 5 calls/engineer/min, with 10x headroom |
| N4 | Ingestion availability | 99.95% | Instrumentation must never fail the request it measures |
| N5 | Cost accuracy | Reconciles to provider invoice to the cent | Anything less and finance rejects the whole system |
| N6 | Data retention | 13 months raw, 7 years aggregated | 13 months gives year-over-year; 7 years is the common financial retention floor |
| N7 | Tenant isolation | Enforced at the database, not only in code | A cross-tenant cost leak is existential for a FinOps product |
| N8 | Recovery | RPO 5 min, RTO 30 min | Usage data is reconstructible from provider bills; the platform is not the system of record |

## 3. Personas

| Persona | Needs | Dashboard | Success |
|---|---|---|---|
| **CFO / VP Finance** | Is spend controlled? What is the trajectory? | Executive | Can defend the AI line item in a board review |
| **FinOps Analyst** | Attribution, chargeback, budget enforcement | Governance | Chargeback reconciles and nobody disputes it |
| **Platform Engineer** | Which prompt, which model, what change | Engineering | Ships a cost fix without a quality regression |
| **Engineering Manager** | Is my team within budget? | Governance | Warned at 80%, not surprised at 110% |
| **AI Architect** | Model selection under constraints | Optimization / Simulator | Justifies a routing decision with numbers |
| **Compliance Officer** | Who used what, under which policy | Governance | Produces an audit trail on request |

## 4. Data model

### Grain

The atomic fact is one metered model interaction. Everything else aggregates it.

```
Organization ─┬─ Department ─┬─ Team ─── User
              │              └─ CostCenter
              ├─ Budget ── Policy ── ApprovalRequest
              ├─ PromptTemplate ── PromptVersion
              └─ UsageEvent ─── CostBreakdown
                     │
                     ├─▶ UsageRollupDaily   (dashboards read this)
                     ├─▶ Anomaly
                     ├─▶ Forecast
                     └─▶ Recommendation ── QualityScore
```

### Physical design

**Partitioning.** `usage_events` is RANGE-partitioned by month on
`occurred_at`. This is the single most consequential physical decision:

- Retention becomes `DROP TABLE` (instant) instead of `DELETE` over 10^9 rows
  (hours of locks, then a VACUUM FULL).
- Every dashboard query is time-bounded, so the planner prunes to one or two
  partitions.
- Per-partition indexes fit in `shared_buffers`; a global index over three
  years does not.

The trade: the partition key must appear in every unique constraint, so the
primary key is `(id, occurred_at)` rather than `id`. This surprises people
during migrations and is documented at the model.

**Rollups over materialized views.** Dashboards read `usage_rollup_daily`,
maintained by incremental upsert. `REFRESH MATERIALIZED VIEW` rewrites the
whole relation — minutes at this volume, and blocking without `CONCURRENTLY`,
which itself requires a unique index and doubles disk. Incremental upsert gives
minute-level freshness at a fraction of the write cost.

**Indexes.** Every composite index leads with `organization_id`, because every
query filters on it and row-level security policies are defined against it.
Partial indexes (`WHERE conversation_id IS NOT NULL`) cover the sparse
correlation columns without paying for the nulls.

### Retention

| Data | Hot | Warm | Cold |
|---|---|---|---|
| Raw usage events | 90 days (Postgres) | 13 months (partitioned) | Dropped |
| Daily rollups | 13 months | 7 years | Object storage, Parquet |
| Anomalies, recommendations | 13 months | 7 years | — |
| Audit logs | 13 months | 7 years (WORM) | — |

## 5. Request paths

### Ingestion (write, high volume)

```
SDK ──▶ POST /ingest/events (202 Accepted)
          │
          ├─ validate, cap tag cardinality
          ├─ dedupe on idempotency_key
          └─ publish to Kafka
                 │
          Consumer group
                 ├─ resolve price (in-process catalog, no DB round trip)
                 ├─ write partitioned fact
                 └─ upsert rollup buckets
```

`202`, not `201`: instrumentation must never couple the caller's latency to our
write path. Price resolution happens in-process against a periodically
refreshed catalog — a per-event database lookup would cap throughput two orders
of magnitude below target.

### Pre-flight enforcement (write, latency-critical)

```
SDK ──▶ POST /governance/preflight
          ├─ estimate cost from the catalog       (in-process)
          ├─ read budget counters                 (Redis, ~1ms)
          ├─ evaluate policies                    (in-process)
          └─ ALLOW | WARN | DOWNGRADE | APPROVE | THROTTLE | BLOCK
```

Redis, not Postgres. The counters are seconds behind the ledger, which is an
accepted trade: blocking at $10,003 instead of exactly $10,000 is operationally
fine; a 40ms tax on every inference call is not. Fails open — see
`services/governance.py`.

### Dashboard queries (read)

```
Browser ──▶ GET /analytics/* ──▶ rollup tables (never raw facts)
```

## 6. Scaling

| Volume | Topology |
|---|---|
| < 1M events/day | Single Postgres, no Kafka, API + worker |
| 1M – 100M/day | Kafka, 3+ consumers, Postgres with read replicas |
| > 100M/day | Partitioned Kafka topics, rollups to a columnar store (ClickHouse), Postgres retains governance state only |

The service layer is stateless and horizontally scalable; the pricing catalog
is process-local and refreshed on an interval, so adding pods adds throughput
linearly until Postgres write capacity becomes the bound — at which point the
columnar migration above is the answer.

## 7. Alternatives considered

| Decision | Alternative | Why rejected |
|---|---|---|
| Price at write, retain inputs | Price at read | Every dashboard becomes a temporal join over a slowly-changing dimension; unviable at 10^9 rows |
| | Price at write only | A rate-card correction can never be applied to history |
| Postgres + partitioning | ClickHouse from day one | Operationally heavier, weaker transactional guarantees for governance state, and Postgres handles the first 100M/day comfortably |
| Kafka for ingestion | Direct writes | No backpressure, no replay, and a database incident becomes an ingestion outage |
| Holt-Winters | Prophet / SARIMA / LSTM | Cost, operability and explainability — see `services/forecasting.py` |
| Scored routing | Learned policy | Cold-start exploration on production traffic, and compliance constraints must not be tradeable |
| Three model layers (domain/ORM/API) | One shared model | Couples the database schema to the public API contract consumed by customer SDKs |
| Decimal money | float, or integer cents | float drifts; integer cents cannot represent sub-cent per-token rates without a second scaling convention |

## 8. Failure modes

| Failure | Behaviour | Rationale |
|---|---|---|
| Policy service unreachable | Pre-flight returns ALLOW | Cost governance must not become a customer-facing outage |
| Pricing catalog stale | Events accepted, flagged `unpriced`, alert raised | Visible failure beats invisible under-reporting |
| Postgres unavailable | Reads degrade, `/health` stays green | Liveness must not restart pods during a database incident |
| Kafka lag | Dashboards go stale, ingestion keeps accepting | Backpressure absorbed by the log, not pushed to the caller |
| Forecast model diverges | MAPE alert at >40% | A silently useless forecast still setting budgets is worse than none |
| Quality baseline missing | Recommendation blocked, not approved | A change cannot be certified against a dimension with no data |
