# Operations

## SLOs

Targets are set where a user notices the difference, not where the graph looks
good.

| Service | SLI | SLO | Error budget (30d) |
|---|---|---|---|
| Ingestion | Successful writes / total | 99.95% | 21 min |
| Pre-flight policy | p99 latency | < 5 ms | — |
| Dashboard queries | p95 latency | < 200 ms | — |
| Dashboard availability | Non-5xx / total | 99.9% | 43 min |
| Data freshness | Ingest → visible, p95 | < 5 min | — |
| Cost accuracy | Reconciliation vs. provider invoice | 100% to the cent | zero |

Two of these deserve explanation.

**Pre-flight at 5ms p99** is not a performance nicety. The SDK calls this before
every inference request. Above ~10ms, teams notice it in their own latency
budget and disable enforcement — at which point every budget and policy in the
product becomes decorative while continuing to render green. This SLO protects
the product's core function, so it pages.

**Cost accuracy has a zero error budget.** A dashboard that is 2% wrong is
worse than no dashboard, because finance will discover the discrepancy during
reconciliation and then distrust every other figure. `aicost_unpriced_events_total`
going non-zero is the leading indicator and files a ticket immediately.

## Deployment

```bash
kubectl apply -f infra/k8s/base.yaml   # namespace, config, secrets, network policy
kubectl apply -f infra/k8s/api.yaml    # API, service, HPA, PDB
kubectl -n aicost rollout status deploy/aicost-api
```

Rollout is `maxUnavailable: 0, maxSurge: 1` — capacity never dips below the
current replica count mid-deploy, so a deployment cannot itself cause the
latency regression people then blame on the release.

`preStop` sleeps 8 seconds before the process exits, covering the window
between endpoint removal and kube-proxy propagation. Without it every rollout
produces a burst of connection resets from pods already removed from the
Service but still receiving traffic.

### Migrations

Migrations run as a Job before the rollout, and must be **backward compatible
with the currently running version** — during a rolling deploy, old and new
pods share a schema. The expand/contract sequence:

1. **Expand** — add the nullable column, backfill, deploy code that writes both
2. **Migrate** — deploy code that reads the new column
3. **Contract** — drop the old column, in a *later* release

Skipping to a rename in one step breaks every pod still running the old
version, for the duration of the rollout.

## Runbooks

### `ApiDown` / `HighErrorRate` (page)

1. `kubectl -n aicost get pods -l app.kubernetes.io/name=aicost-api`
2. If pods are `CrashLoopBackOff`, check logs for a config error — most often a
   missing secret after a rotation.
3. If pods are healthy but erroring, check Postgres and Redis reachability.
   The API degrades rather than dies when Postgres is unavailable; total 5xx
   usually means a config or dependency-resolution failure, not load.
4. Roll back: `kubectl -n aicost rollout undo deploy/aicost-api`.

### `PolicyEvaluationSlow` (page)

The pre-flight path is on customer inference latency. Treat as urgent.

1. Check Redis latency: `redis-cli --latency-history`.
2. Check the budget-counter cache hit rate. A cold cache after a Redis restart
   forces Postgres fallback, which is 10-40x slower.
3. **Mitigation:** set `POLICY_FAIL_OPEN=true` (the default) — requests are
   allowed rather than delayed. Enforcement accuracy degrades; customer latency
   does not.
4. Warm the counters by replaying the current period's rollups.

### `UnpricedEventsDetected` (ticket)

Spend is being under-reported *right now*, silently.

1. Identify the model: the alert carries `provider` and `model` labels.
2. Run the pricing sync: `celery -A app.workers.celery_app call
   app.workers.celery_app.sync_provider_pricing`.
3. If the model is genuinely new, add a rate card with `effective_from` set to
   **the first time the model was seen**, not now — otherwise the backfill
   leaves a zero-cost gap.
4. Re-price the affected window; the cost engine is a pure function, so a
   bounded backfill over the partition is deterministic.

### `IngestionStalled` (page)

1. Confirm whether it is us or them: check whether *any* tenant is sending.
   A single tenant stopping is their deploy; all tenants stopping is us.
2. Check Kafka consumer lag and the ingest API's error rate.
3. Check for an auth failure — an expired API key mass-rotation shows as a
   clean stop with 401s rather than errors.
4. Events are buffered by the SDK and by Kafka; a short stall backfills on
   recovery without data loss. Do not restart consumers to "unstick" them
   before checking lag, which usually makes it worse.

### `SpendRateSpike` (page)

1. Open the anomaly dashboard — findings are ranked by dollar impact, so the
   top row is the driver.
2. Most common causes in order: a runaway agent, a retry storm against a
   degraded provider, or a newly deployed batch job with no cost ceiling.
3. Immediate containment: set the offending scope's budget
   `action_at_limit: block` with a `hard_stop_multiplier`. This takes effect on
   the next pre-flight check, within seconds.
4. Then fix the cause — a blocked budget is containment, not a resolution.

### `ForecastAccuracyDegraded` (ticket)

MAPE above 40% means budgets are being set from an unreliable projection.

1. Check for a structural break — a launch, a migration, a large one-off batch.
   Holt-Winters cannot see a level shift coming and takes ~2 seasonal cycles to
   absorb one.
2. Verify the history series is dense and excludes the current partial day.
   A partial final day is the single most common cause of inflated MAPE; it
   biases the level down and the backtest scores against the truncated value.
3. If the break is genuine and permanent, truncate history to the post-break
   window rather than waiting for the model to forget.

## Capacity

| Load | API pods | Worker pods | Postgres |
|---|---|---|---|
| < 1M events/day | 3 | 1 | db.r6g.large |
| 10M/day | 6 | 3 | db.r6g.2xlarge + 1 replica |
| 100M/day | 15 | 8 | db.r6g.8xlarge + 2 replicas, monthly partitions |
| > 100M/day | 30 | 16 | Migrate rollups to ClickHouse |

The HPA scales up fast (30s stabilisation) and down slowly (600s). Asymmetry is
intentional: the cost of a few idle pods is trivial next to an under-provisioned
service meeting the next spike.

## Backup and recovery

| Aspect | Value |
|---|---|
| RPO | 5 minutes (WAL archiving) |
| RTO | 30 minutes (PITR restore) |
| Retention | 35 days PITR, 7 years for aggregates |
| Rehearsal | Quarterly, timed, into a scratch environment |

Restores are rehearsed on a schedule because a backup that has never been
restored is a hypothesis, not a control.

## Production readiness checklist

### Before first production traffic

- [ ] `SECRET_KEY` sourced from the secret store; startup guard verified to
      reject the development default
- [ ] `DOCS_ENABLED=false` — the OpenAPI schema is free reconnaissance
- [ ] OIDC configured; local password auth confirmed disabled
- [ ] Row-level security policies applied and verified with a cross-tenant
      read attempt
- [ ] `audit_logs` UPDATE grant revoked for the application role
- [ ] Network policies applied; egress to `169.254.169.254` confirmed blocked
- [ ] Partitions created 3+ months ahead
- [ ] Pricing catalog synced and reconciled against one real provider invoice
- [ ] Alert routing tested end to end — including that a page actually pages
- [ ] Restore rehearsed from a real backup
- [ ] Load test at 2x projected peak
- [ ] Pre-flight p99 measured under load, not just at idle

### Ongoing

- [ ] Weekly: reconcile platform cost against provider invoices
- [ ] Weekly: review `aicost_recommendation_accuracy_ratio` — are our estimates honest?
- [ ] Monthly: review forecast MAPE per scope
- [ ] Monthly: review attribution coverage; drive unattributed spend down
- [ ] Quarterly: restore rehearsal, access review, dependency audit

## What to watch that is not an alert

Three signals that indicate the platform is losing its usefulness long before
anything breaks:

1. **Attribution coverage trending down.** New services are shipping without
   instrumentation. Unattributed spend cannot be charged back, and the
   chargeback statement quietly becomes fiction.
2. **Recommendation accuracy drifting below 1.0.** We are over-promising.
   Caught early it is a modelling fix; caught late it is a credibility problem
   that no amount of subsequent accuracy repairs.
3. **Quick wins never being applied.** If config-only, zero-risk
   recommendations sit open for months, the product is producing analysis
   nobody acts on — which is a product problem, not an engineering one.
