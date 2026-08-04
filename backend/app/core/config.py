"""Application configuration.

Settings come from environment variables only — never a checked-in file. This
is a hard requirement for the twelve-factor deployment model the Kubernetes
manifests assume, and it is what lets the same image promote unchanged from
dev to prod with only the ConfigMap/Secret differing.

Secrets (`SECRET_KEY`, database password, provider API keys) are injected from
the cluster secret store (External Secrets Operator -> AWS Secrets Manager /
Azure Key Vault), never baked into the image or the ConfigMap.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # -- application --------------------------------------------------------
    app_name: str = "AI Cost Intelligence Platform"
    environment: Literal["local", "dev", "staging", "production"] = "local"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    docs_enabled: bool = True

    # -- security -----------------------------------------------------------
    secret_key: str = Field(default="dev-only-insecure-key-change-me", min_length=16)
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 30
    refresh_token_ttl_days: int = 7
    #: OIDC/SAML discovery for enterprise SSO. When set, local password auth is
    #: disabled — mixed auth modes are a standing audit finding.
    oidc_issuer: str | None = None
    oidc_client_id: str | None = None
    oidc_client_secret: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # -- database -----------------------------------------------------------
    database_url: str = "postgresql+asyncpg://aicost:aicost@localhost:5432/aicost"
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    #: Recycle below typical cloud load-balancer idle timeouts (usually 350s)
    #: so the pool never hands out a connection the network has already killed.
    db_pool_recycle: int = 300
    db_echo: bool = False

    # -- redis / cache ------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    #: Budget counters live here; see governance.py on why the pre-flight path
    #: cannot touch Postgres.
    budget_counter_ttl_seconds: int = 172_800
    pricing_refresh_seconds: int = 300

    # -- ingestion ----------------------------------------------------------
    kafka_bootstrap_servers: str | None = None
    kafka_usage_topic: str = "aicost.usage.v1"
    ingest_batch_size: int = 500
    ingest_max_batch_wait_ms: int = 200
    #: Reject events older than this. Late data past the window would land in
    #: already-closed accounting periods and silently restate finished reports.
    max_event_age_hours: int = 72

    # -- workers ------------------------------------------------------------
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # -- observability ------------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    otel_exporter_endpoint: str | None = None
    otel_service_name: str = "aicost-api"
    metrics_enabled: bool = True
    #: Head-based sampling. 100% tracing on a service handling 50k events/sec
    #: costs more than the platform saves.
    trace_sample_ratio: float = 0.05

    # -- governance defaults ------------------------------------------------
    default_budget_alert_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.8, 0.95]
    )
    #: Fail open if the policy engine is unavailable. See governance.py.
    policy_fail_open: bool = True
    policy_evaluation_timeout_ms: int = 50

    # -- rate limiting ------------------------------------------------------
    rate_limit_per_minute: int = 600
    ingest_rate_limit_per_minute: int = 60_000

    # -- demo ---------------------------------------------------------------
    #: Populate the in-memory store with synthetic usage at startup so the
    #: platform is explorable with no external services. Refused in production
    #: regardless of this value; see the startup guard in main.py.
    seed_demo_data: bool = False
    seed_demo_days: int = 90

    @field_validator("secret_key")
    @classmethod
    def _reject_default_secret_in_production(cls, value: str, info) -> str:  # type: ignore[no-untyped-def]
        """Refuse to boot production with the development secret.

        A startup crash is dramatically preferable to a production deployment
        signing tokens with a key that is published in the repository.
        """
        env = (info.data or {}).get("environment")
        if env == "production" and value.startswith("dev-only"):
            raise ValueError("SECRET_KEY must be set to a real secret in production")
        return value

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def sso_enabled(self) -> bool:
        return bool(self.oidc_issuer and self.oidc_client_id)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. Settings are immutable for the process lifetime."""
    return Settings()
