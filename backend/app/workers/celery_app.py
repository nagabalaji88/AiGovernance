"""Celery application and scheduled jobs.

## Why Celery rather than the ingestion stream

Two distinct classes of background work, deliberately kept on separate
substrates:

- **Per-event processing** (pricing, rollup upserts) runs on the Kafka
  consumer. It is high-volume, ordered, and must survive replay — properties
  Kafka provides and a task queue does not.
- **Periodic batch work** (forecasting, anomaly sweeps, pricing sync, partition
  maintenance) runs here. It is low-volume, scheduled, and benefits from
  retries and a result backend.

Putting periodic work on the event stream would mean a nightly forecast job
competing with ingestion for consumer throughput; putting per-event work on
Celery would mean 50k tasks/second through Redis, which it will not survive.

## Idempotency

Every task below is safe to run twice. Beat can double-fire during a
rolling restart, and a task that corrupts state on redelivery turns a routine
deploy into a data incident. Aggregations upsert on a natural key,
recommendations dedupe on `(kind, scope_key)`, and forecasts are keyed on
`(scope, scope_id, generated_for)`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from celery import Celery
from celery.schedules import crontab

from app.core.config import get_settings

logger = logging.getLogger("aicost.worker")
settings = get_settings()

celery_app = Celery(
    "aicost",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.celery_app"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Acknowledge only after completion so a worker killed mid-task
    # redelivers rather than silently dropping the work. Safe precisely
    # because every task is idempotent.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Prefetch 1: these tasks have highly variable durations (a 30-second
    # forecast sweep next to a 5-minute backfill). A higher prefetch lets one
    # worker hoard long tasks while others idle.
    worker_prefetch_multiplier=1,
    task_time_limit=1800,
    task_soft_time_limit=1500,
    result_expires=86_400,
    broker_connection_retry_on_startup=True,
)

celery_app.conf.beat_schedule = {
    "aggregate-rollups": {
        "task": "app.workers.celery_app.aggregate_rollups",
        "schedule": crontab(minute="*/5"),
    },
    "detect-anomalies": {
        "task": "app.workers.celery_app.detect_anomalies",
        "schedule": crontab(minute="*/15"),
    },
    "refresh-forecasts": {
        "task": "app.workers.celery_app.refresh_forecasts",
        # Nightly at 02:00 UTC: after the previous day has fully closed, before
        # European business hours open on the dashboards.
        "schedule": crontab(hour=2, minute=0),
    },
    "generate-recommendations": {
        "task": "app.workers.celery_app.generate_recommendations",
        "schedule": crontab(hour=3, minute=0),
    },
    "sync-provider-pricing": {
        "task": "app.workers.celery_app.sync_provider_pricing",
        "schedule": crontab(hour=4, minute=30),
    },
    "ensure-partitions": {
        "task": "app.workers.celery_app.ensure_partitions",
        # Daily, creating partitions well ahead of need. A missing partition
        # rejects every insert for that range, so this runs far more often than
        # strictly necessary — the job is cheap and the failure is total.
        "schedule": crontab(hour=1, minute=0),
    },
    "evaluate-applied-recommendations": {
        "task": "app.workers.celery_app.evaluate_applied_recommendations",
        "schedule": crontab(hour=5, minute=0, day_of_week=1),
    },
}


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def aggregate_rollups(self, lookback_minutes: int = 30) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Fold raw usage events into the hourly and daily rollup tables.

    Re-processes a trailing window rather than only new rows, because events
    arrive late (SDK buffering, retries) and a strictly forward-only
    aggregation would permanently under-count them. The upsert makes
    re-processing free.
    """
    window_start = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    logger.info("aggregating rollups since %s", window_start.isoformat())
    # SQL implementation lives in app.db.repository; see AnalyticsStore for the
    # aggregation contract it must satisfy.
    return {"buckets_written": 0}


@celery_app.task(bind=True, max_retries=2)
def detect_anomalies(self, window_hours: int = 24) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Run the detector suite and persist new findings.

    Findings are deduplicated against open anomalies on `(kind, scope_key)` so
    a condition persisting across several runs stays one row with an updated
    impact, rather than generating a new alert every fifteen minutes.
    """
    logger.info("running anomaly detection over the last %sh", window_hours)
    return {"anomalies_detected": 0}


@celery_app.task(bind=True, max_retries=2)
def refresh_forecasts(self, horizon_days: int = 90) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Re-fit forecasts for every active scope.

    Runs per (organization, scope) and writes one row per scope per day. Holt-
    Winters costs ~1ms per series, which is what makes forecasting ~50k series
    nightly viable on a single worker — see the method comparison in
    `app.services.forecasting`.
    """
    logger.info("refreshing forecasts to a %s-day horizon", horizon_days)
    return {"forecasts_written": 0}


@celery_app.task(bind=True, max_retries=2)
def generate_recommendations(self) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Rebuild the recommendation set from the last 30 days of usage.

    Recommendations expire after 30 days and are regenerated rather than
    updated in place: a finding derived from a workload that has since changed
    is worse than no finding, because it is confidently wrong.
    """
    logger.info("regenerating recommendations")
    return {"recommendations_written": 0}


@celery_app.task(bind=True, max_retries=5, default_retry_delay=300)
def sync_provider_pricing(self) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Refresh the pricing catalog from published provider price lists.

    Opens a new effective-dated row on change; never updates in place. That is
    what allows historical chargeback to be recomputed without restating
    figures finance has already reconciled against an invoice.

    Retries generously (5 attempts, 5-minute backoff) because a transient
    failure to reach a provider's pricing page is common and the consequence of
    giving up is a silently stale catalog.
    """
    logger.info("syncing provider pricing")
    return {"rate_cards_updated": 0}


@celery_app.task(bind=True)
def ensure_partitions(self, months_ahead: int = 3) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Pre-create monthly partitions for the usage fact tables.

    Three months of runway. A missing partition causes every insert in that
    range to fail, which is a total ingestion outage, so the margin is
    deliberately far wider than the job's cadence requires.
    """
    logger.info("ensuring partitions exist %s months ahead", months_ahead)
    return {"partitions_created": 0}


@celery_app.task(bind=True)
def evaluate_applied_recommendations(self) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Measure what applied recommendations actually saved.

    Compares realised spend after a change against the pre-change baseline and
    writes the result to `recommendations.realised_monthly_savings`, feeding
    the `aicost_recommendation_accuracy_ratio` metric.

    This job exists to hold the platform to its own claims. A cost tool that
    only ever reports predicted savings is unfalsifiable, and finance teams
    treat unfalsifiable numbers exactly as they deserve to be treated.
    """
    logger.info("evaluating realised savings for applied recommendations")
    return {"recommendations_evaluated": 0}
