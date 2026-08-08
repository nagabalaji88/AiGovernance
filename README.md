# AI Cost Intelligence & Token Optimization Platform

Measure, forecast, attribute and optimize enterprise AI spend across fifteen
providers — without degrading response quality.

Enterprise LLM spend is unpredictable because the cost drivers are invisible:
prompts grow silently, conversation history is resent verbatim every turn, RAG
pipelines inject context nobody reads, agents fail to terminate, and retries
burn money that returns nothing. This platform makes each of those visible,
prices it, and tells you what to change — with a quality gate that can veto any
recommendation that would make answers worse.

---

## Requirements

Python **3.9 – 3.13** (CI tests both ends of that range) and Node 22.

> **Note on Python 3.9:** 3.9 reached end-of-life in October 2025 and no longer
> receives security patches. It is supported here because the deployment target
> requires it, but for a platform making SOC 2 / ISO 27001 claims the
> interpreter itself is part of the audit surface. Moving the floor to 3.11
> would also let the `Optional[...]` annotations revert to `X | None` and drop
> the `StrEnum` backport — see `docs/ROADMAP.md`.

## Run it

No database, no broker, no configuration:

```bash
make install
make demo
```

- Dashboards: <http://127.0.0.1:5173>
- API docs: <http://127.0.0.1:8000/docs>

The demo seeds ~72,000 synthetic usage events across five departments and nine
workloads, deliberately containing the pathologies the platform exists to find:
a chatbot whose context grows every turn, an agent that never terminates, a
retry storm, a FAQ endpoint bypassing cache, and a RAG pipeline with a 4k-token
static preamble and no prompt caching.

The full production topology — Postgres, Redis, Celery, Prometheus, Grafana —
runs with `make up`.

---

## What it does

| Capability | What it answers |
|---|---|
| **Cost engine** | What did this request actually cost, against the rate card in force *at the time*? |
| **Token analytics** | Which part of the prompt spent the money — system preamble, few-shot, tools, RAG, or history? |
| **Forecasting** | What will we spend, and when does this budget run dry? |
| **Anomaly detection** | What is burning money right now that shouldn't be? |
| **Prompt optimization** | Which tokens in this prompt are removable, and what is that worth? |
| **Model routing** | What is the cheapest model that will still do this job? |
| **RAG tuning** | What are `top_k`, `chunk_size` and overlap actually costing? |
| **Cache advisory** | Which cache tier applies here, at what TTL, worth how much? |
| **Simulation** | What would this change cost *before* I make it? |
| **Governance** | Budgets, policies, chargeback, approvals, audit. |
| **Quality gate** | Did that optimization make the answers worse? |

---

## Design decisions worth knowing

Each of these is argued in full in the module it governs; the short version:

**Money is `Decimal`, never `float`.** A tenant aggregates 10^8 sub-cent
amounts into invoices that must reconcile to the cent. Binary floating point
drifts under summation and, worse, drifts *differently* depending on shard
ordering, so two recomputations of the same month disagree. → `domain/money.py`

**Rate cards are effective-dated and never mutated.** Providers change prices
without warning. A mutable price column means re-running last quarter's
chargeback silently restates numbers finance already paid against. Cost
resolution keys on the timestamp of the *event*, never "now". →
`domain/pricing.py`

**Cached tokens are a separate billable class, not a discount.** Providers
meter cache reads as a disjoint bucket from fresh input. Treating them as a
discount on the input count mis-prices cache-heavy workloads by up to 90%. →
`domain/usage.py`

**Unknown models raise instead of pricing at zero.** Silently zero-costing an
unrecognised model is how a platform under-reports spend on exactly the newest,
most expensive model a team just adopted. Events are accepted, flagged
`unpriced`, and alerted on. → `services/cost_engine.py`

**Forecasting is Holt-Winters, not Prophet or an LSTM.** We forecast ~50k
series nightly. Prophet costs ~2s per series (28 hours of compute); an LSTM
adds GPU and MLOps burden and is uninspectable. Damped Holt-Winters runs in
~1ms, has no native dependencies, and a FinOps analyst can be shown the
level/trend/season decomposition and disagree with it. A forecast finance
doesn't trust doesn't get used. → `services/forecasting.py`

**Anomaly detection uses median/MAD, not mean/σ.** Cost series contain the very
outliers being hunted; one $50k runaway job inflates σ enough to hide the next
one. MAD stays stable until half the observations are anomalous. Findings rank
by *dollar impact*, not statistical surprise. → `services/anomaly.py`

**Routing is a scored linear utility, not a learned policy.** A bandit explores
by deliberately routing production traffic to bad models — an unacceptable
first week. And compliance constraints (residency, approved vendors) are hard
filters applied *before* scoring, so no weight can trade them away. →
`services/model_router.py`

**Simulated levers compose on tokens, not sum on cost.** 30% compression plus a
50% cheaper model is a 65% saving, not 80%. Summing deltas overstates
compounding scenarios and can claim >100%. → `services/simulator.py`

**Quality has veto power.** Every other engine proposes spending less; this one
blocks changes that regress a guarded dimension, and shows blocked
recommendations *with the reason* rather than hiding them. Hiding them just
means teams implement the change without the gate. → `services/quality.py`

**Prompt analysis never retains prompt text.** Findings carry counts, offsets
and hashes only — no content — which keeps the platform outside the tenant's
PII and data-residency scope while still proving savings. →
`services/prompt_optimizer.py`

**Governance enforces pre-flight and fails open.** The SDK checks before
dispatching, against Redis counters, with a 5ms p99 budget. Slower and teams
disable it, at which point the platform governs nothing. If the policy service
is unreachable it returns ALLOW: a failure in cost governance must never become
an outage in the customer's product. → `services/governance.py`

---

## Architecture

```
   SDKs / proxies ──▶ Ingest API ──▶ Kafka ──▶ Consumers ──▶ Postgres
        │              (202)                   (price,        (partitioned
        │                                       attribute,     facts +
        └──▶ Pre-flight policy check            rollup)        rollups)
             (Redis counters, 5ms p99)                            │
                                                                  ▼
   Dashboards ◀── Query API ◀── rollup tables        Celery beat ──┤
   (React)        (FastAPI)                          forecast,     │
                                                     anomaly,      │
                                                     recommend ────┘
```

Dashboards never touch the raw fact table. They read pre-aggregated rollups
maintained incrementally by the consumers; raw events exist for drill-down and
deterministic re-pricing.

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Repository layout

```
backend/
  app/
    domain/        Pure business objects — money, pricing, usage, enums
    services/      The engines. No I/O, no framework, exhaustively testable
    api/           FastAPI routes, schemas, dependencies
    db/            SQLAlchemy models, partitioning strategy
    workers/       Celery tasks and beat schedule
    store.py       In-memory analytics store (the repository contract)
  tests/           210 tests
frontend/
  src/
    pages/         Five role-specific dashboards
    components/    Charts and the glass design system
    lib/           Typed API client, design tokens
infra/
  k8s/             Kubernetes manifests
  observability/   Prometheus scrape config and alert rules
docs/              Architecture, API, security, operations, roadmap
```

---

## Testing

```bash
make test     # 210 tests with coverage
make check    # everything CI runs
```

Tests concentrate where correctness is load-bearing: costing precision under
10^5 aggregations, effective-dated pricing, forecast degradation paths,
routing constraint enforcement, quality-gate blocking, and API contracts.

---

## Documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Requirements, personas, HLD/LLD, data model, scaling |
| [API.md](docs/API.md) | Endpoint reference, auth, pagination, errors |
| [SECURITY.md](docs/SECURITY.md) | Threat model, RBAC, compliance mapping |
| [OPERATIONS.md](docs/OPERATIONS.md) | Deployment, SLOs, runbooks, readiness checklist |
| [ROADMAP.md](docs/ROADMAP.md) | Sprint plan, risk register, future work |

---

## Status

The engines, API, dashboards, tests and infrastructure are complete and
runnable. The persistence layer is defined (SQLAlchemy models, partitioning,
Alembic) with the in-memory store implementing the same query contract; the
SQL repository binding is the remaining integration step, isolated behind one
dependency provider in `api/deps.py`. Container images and Kubernetes manifests
are written and validated but have not been built in this environment (no
Docker daemon available).
