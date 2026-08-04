# Delivery plan, risks and roadmap

## Sprint plan

Two-week sprints. Sequencing follows one rule: **nothing is built before the
thing that makes it trustworthy.** Recommendations before quality measurement
produce advice nobody should follow; forecasting before accurate costing
produces confident projections of wrong numbers.

### Phase 1 — Measure (Sprints 1-3)

Without accurate measurement nothing downstream is defensible.

| Sprint | Delivered | Exit criteria |
|---|---|---|
| 1 | Domain model, effective-dated pricing, cost engine, token accounting | Costs reconcile to the cent against a real provider invoice |
| 2 | Ingestion API, idempotency, Kafka pipeline, partitioned schema, rollups | 50k events/sec sustained; freshness p95 < 5 min |
| 3 | Token analytics, attribution, executive + engineering dashboards | A team can see which prompt region spends their money |

**Gate:** a finance stakeholder agrees the numbers match their invoice. If this
fails, everything after it is built on sand — do not proceed.

### Phase 2 — Understand (Sprints 4-6)

| Sprint | Delivered | Exit criteria |
|---|---|---|
| 4 | Forecasting, budget exhaustion projection, MAPE backtesting | MAPE < 25% on 30-day horizons for stable workloads |
| 5 | Anomaly detection, waste quantification, alerting | < 10 alerts/day/tenant; every alert carries a dollar figure |
| 6 | Budgets, policy engine, pre-flight enforcement, RBAC, audit | Pre-flight p99 < 5ms under load |

**Gate:** alert volume is low enough that people read them. An ignored alerting
system has negative value — it trains people to ignore the real one.

### Phase 3 — Optimize (Sprints 7-10)

| Sprint | Delivered | Exit criteria |
|---|---|---|
| 7 | Quality measurement, deterministic scorers, quality gate | Gate demonstrably blocks a known-bad change |
| 8 | Prompt optimizer, cache advisor, RAG optimizer | First $10k/yr saving verified in production |
| 9 | Model router, simulation engine, what-if UI | A routing decision survives an architect's challenge |
| 10 | Recommendation engine, priority ranking, realised-savings tracking | Realised savings within 25% of predicted |

**Gate:** quality ships *before* the optimizers it governs. Reversing this order
is how a cost programme causes an incident and gets cancelled.

### Phase 4 — Govern and scale (Sprints 11-13)

| Sprint | Delivered | Exit criteria |
|---|---|---|
| 11 | Chargeback, showback, approvals, compliance reports | Chargeback accepted by finance without dispute |
| 12 | SSO, multi-tenancy hardening, RLS, SOC 2 evidence | Cross-tenant read attempt fails at the database |
| 13 | Load testing, DR rehearsal, runbooks, production readiness | 2x peak sustained; restore rehearsed and timed |

## Risk register

Ordered by expected damage, not probability.

| # | Risk | L | I | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | **Optimization degrades quality; programme is discredited** | M | Critical | Quality gate with veto power, shipped before the optimizers; guarded dimensions require a baseline; blocked items shown with reasons | AI Lead |
| R2 | **Cost figures don't reconcile with provider invoices** | M | Critical | Decimal arithmetic end to end; effective-dated cards; unpriced events alert rather than zero-cost; weekly reconciliation | Eng Lead |
| R3 | **Cross-tenant data leak** | L | Critical | RLS at the database, `organization_id` leading every index, tenancy from the token never a parameter, penetration test before GA | Security |
| R4 | **Pre-flight latency causes teams to disable enforcement** | M | High | 5ms p99 SLO that pages; Redis counters never Postgres; fail-open; latency measured under load not idle | Platform |
| R5 | **Provider pricing changes silently** | H | High | Daily sync opening effective-dated rows; unpriced-event alerting; reconciliation catches drift within a week | Platform |
| R6 | **Alert fatigue makes anomaly detection worthless** | H | Medium | Severity gated on dollar impact not just sigma; findings rolled up per feature; < 10/day/tenant target | Eng Lead |
| R7 | **Instrumentation coverage stalls; attribution stays incomplete** | H | Medium | Coverage is a first-class reported metric; unattributed spend allocated proportionally rather than absorbed, so under-instrumented teams still pay | FinOps |
| R8 | **Forecast drifts after a structural break** | M | Medium | MAPE monitored per scope, alert above 40%; damped trend limits runaway extrapolation; history truncation runbook | Data |
| R9 | **Recommendation savings over-promised** | M | Medium | Realised savings measured post-application and published as a metric; de-overlapped savings; conservative confidence | Product |
| R10 | **Postgres becomes the write bottleneck** | M | Medium | Partitioning, rollups, read replicas; ClickHouse migration path defined at 100M events/day | Platform |
| R11 | **Semantic cache returns a wrong answer** | L | High | 0.97 default threshold, per-prompt opt-in, shadow-run before serving, never recommended for entity-varying queries | AI Lead |
| R12 | **Key-person dependency on the costing logic** | M | Low | Engines are pure functions with decision records in-module; 176 tests document intended behaviour | Eng Lead |

### Risks explicitly accepted

Stated so they are decisions rather than oversights:

- **Budget enforcement can overshoot by the counter lag.** Blocking at $10,003
  instead of $10,000 is acceptable; a database round trip on the inference hot
  path is not.
- **Fail-open governance allows spend during a policy outage.** A governance
  failure must not become a customer-facing outage.
- **Semantic paraphrase detection in prompts is weaker than embeddings would
  give.** The privacy property (never ingesting prompt text) is worth more than
  the detection accuracy.
- **Token estimates are approximate when the SDK cannot supply exact counts.**
  Flagged as approximate wherever surfaced rather than silently presented as
  exact.

## Future work

### Near term (next two quarters)

**Natural-language cost queries.** "What did the support team spend on GPT-5
last month, and why did it jump?" — translated to a bounded query over the
rollup tables. Deliberately *not* free-form SQL generation: a text-to-SQL
surface over a multi-tenant cost database is a data-exfiltration primitive.
Constrained to a parameterised query grammar with the tenant filter applied
server-side.

**Autonomous optimization with approval workflow.** The platform already knows
which changes are config-only and zero-risk. The next step is proposing them as
pull requests against the tenant's own configuration repository — with a human
approving the merge, never applying changes to production systems directly.

**Continuous prompt A/B testing.** `PromptComparison` exists; wiring it to
automatic traffic splitting with a statistical stopping rule closes the loop
from "this prompt is 40% cheaper" to "this prompt is 40% cheaper and measurably
no worse".

**Carbon accounting.** Energy cost is already modelled for self-hosted
inference. Extending it to a gCO2e figure per request is mostly a matter of
regional grid intensity data, and it is increasingly a procurement requirement
rather than a nice-to-have.

### Medium term

**Learned routing, safely.** Replace the linear utility with a contextual
bandit — but only after enough measured quality data exists to initialise it,
and with compliance constraints remaining hard filters outside the policy. The
cold-start problem is why this is not in v1, not a limitation of the approach.

**Per-customer unit economics.** Cost per customer, per transaction, per
support ticket resolved. The attribution model already carries `customer_id`;
what is missing is joining it to a revenue signal to produce a gross-margin
view of AI spend.

**Cross-provider fungibility index.** Given a workload, how substitutable is
the current provider — measured, not assumed. This is what turns a renewal
negotiation from a hope into a position.

**Columnar analytics tier.** ClickHouse for rollups above 100M events/day, with
Postgres retaining governance state where transactional guarantees matter.

### Deliberately not planned

- **A proxy that sits in the inference path.** Several competitors do this. It
  gives richer data and makes the platform a hard dependency for every AI call
  in the enterprise — an availability liability we are not willing to be. The
  SDK reports; it does not intermediate.
- **Storing prompts and completions.** The privacy boundary is the product's
  most valuable non-functional property. Crossing it would unlock better
  analysis and would put the platform inside every customer's PII scope.
- **Automatic model switching without human approval.** The router recommends.
  A system that silently changes which model answers a customer's question,
  based on a cost heuristic, will eventually do so at the worst possible moment.
