"""Metrics and structured logging.

## Metric design

Two rules govern every metric here:

1. **Bounded label cardinality.** Labels are route templates, providers, model
   names and status classes — all small, enumerable sets. Never a user id,
   request id, prompt fingerprint or customer id. Each unique label
   combination is a separate time series in Prometheus; an unbounded label
   turns a metrics backend into an outage. This is the single most common way
   teams take down their own monitoring.

2. **Business metrics alongside technical ones.** `aicost_spend_usd_total`
   sits next to `http_request_duration_seconds` so a single Grafana board can
   answer "did that deploy make us slower *or* more expensive". Cost is a
   first-class operational signal in this product, not an afterthought.

Histogram buckets are chosen to bracket the SLOs rather than using the client
library defaults: the default buckets top out at 10s, which tells you nothing
useful about a 200ms p95 target.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:
    from app.core.config import Settings

# -- HTTP ---------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "http_requests_total",
    "HTTP requests by method, route template and status code.",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "path"],
    # Bracketed around the 200ms p95 dashboard SLO, with resolution below it
    # so a regression from 80ms to 150ms is visible rather than being absorbed
    # into one wide bucket.
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 8.0),
)

# -- Ingestion ----------------------------------------------------------------

EVENTS_INGESTED = Counter(
    "aicost_events_ingested_total",
    "Usage events accepted, by provider and outcome.",
    ["provider", "outcome"],
)

INGEST_LAG = Histogram(
    "aicost_ingest_lag_seconds",
    "Delay between an event occurring at the client and being persisted.",
    # Wide and log-scaled: SDK buffering means legitimate lag spans seconds to
    # tens of minutes, and the interesting signal is the shape of that tail.
    buckets=(1, 5, 15, 60, 300, 900, 3600, 21600, 86400),
)

UNPRICED_EVENTS = Counter(
    "aicost_unpriced_events_total",
    "Events accepted with no matching rate card. Any sustained non-zero rate "
    "means the pricing catalog is stale and spend is being under-reported.",
    ["provider", "model"],
)

# -- Business -----------------------------------------------------------------

SPEND_TOTAL = Counter(
    "aicost_spend_usd_total",
    "Cumulative resolved spend in USD.",
    ["provider", "model", "environment"],
)

TOKENS_TOTAL = Counter(
    "aicost_tokens_total",
    "Cumulative tokens by billable class.",
    ["provider", "model", "token_class"],
)

WASTED_SPEND = Counter(
    "aicost_wasted_spend_usd_total",
    "Spend on failed, retried or cancelled requests.",
    ["provider", "reason"],
)

CACHE_SAVINGS = Counter(
    "aicost_cache_savings_usd_total",
    "Realised savings attributed to caching.",
    ["tier"],
)

BUDGET_UTILISATION = Gauge(
    "aicost_budget_utilisation_ratio",
    "Current period spend as a fraction of the budget.",
    ["scope", "scope_id"],
)

# -- Governance ---------------------------------------------------------------

POLICY_DECISIONS = Counter(
    "aicost_policy_decisions_total",
    "Pre-flight policy decisions by action taken.",
    ["action"],
)

POLICY_LATENCY = Histogram(
    "aicost_policy_evaluation_seconds",
    "Pre-flight policy evaluation latency. SLO: p99 < 5ms.",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1),
)

ANOMALIES_DETECTED = Counter(
    "aicost_anomalies_detected_total",
    "Anomalies raised, by kind and severity.",
    ["kind", "severity"],
)

RECOMMENDATIONS_OPEN = Gauge(
    "aicost_recommendations_open",
    "Open recommendations and the annualised savings they represent.",
    ["kind"],
)

#: Accuracy of our own estimates, tracked openly. When a recommendation is
#: applied we measure the realised saving and compare it to what we predicted.
#: Publishing this makes the platform's claims falsifiable, which is what makes
#: them credible to a finance organisation.
RECOMMENDATION_ACCURACY = Histogram(
    "aicost_recommendation_accuracy_ratio",
    "Realised savings divided by predicted savings for applied recommendations.",
    buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 5.0),
)

FORECAST_ERROR = Gauge(
    "aicost_forecast_mape",
    "Backtested mean absolute percentage error of the active forecast model.",
    ["scope"],
)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


def configure_logging(settings: Settings) -> None:
    """Configure structured logging.

    JSON in every deployed environment so log aggregation can index fields
    rather than regex-parsing prose; human-readable locally where a developer
    is reading with their eyes. Uvicorn's access log is silenced because the
    request middleware already emits a richer, correlated line and duplicate
    access logs double log volume for no added signal.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    if settings.log_format == "json":
        try:
            import structlog

            structlog.configure(
                processors=[
                    structlog.contextvars.merge_contextvars,
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso", utc=True),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ],
                wrapper_class=structlog.make_filtering_bound_logger(level),
                cache_logger_on_first_use=True,
            )
        except ImportError:  # pragma: no cover - structlog is a hard dep in prod
            pass

    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False
