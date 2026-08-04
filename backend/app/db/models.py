"""SQLAlchemy models.

## Partitioning strategy

`usage_events` and `usage_costs` are declaratively partitioned by RANGE on
`occurred_at`, one partition per month, created ahead of time by a scheduled
job. This is the single most important physical design decision in the schema:

- **Retention becomes metadata.** Dropping a month of raw events is
  `DROP TABLE`, which is instant, rather than a `DELETE` over 10^9 rows that
  would hold locks for hours and leave the table needing a VACUUM FULL.
- **Queries prune.** Every dashboard query is time-bounded, so the planner
  touches one or two partitions instead of the whole history.
- **Indexes stay in memory.** A per-partition index on a month of data fits in
  shared_buffers; a global index over three years does not.

## Rollup tables

Dashboards never query `usage_events` directly. They read from
`usage_rollup_hourly` / `usage_rollup_daily`, maintained incrementally by the
aggregation worker. The raw table is for drill-down and re-pricing only.

The alternative — a materialized view refreshed on a schedule — was rejected
because `REFRESH MATERIALIZED VIEW` rewrites the whole thing, which at this
volume takes minutes and blocks concurrent reads without CONCURRENTLY (which in
turn requires a unique index and doubles disk usage). Incremental upsert into a
plain table gives us minute-level freshness at a fraction of the write cost.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _now() -> datetime:
    return datetime.now(UTC)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Tenancy and identity
# ---------------------------------------------------------------------------


class Organization(Base, TimestampMixin):
    """The tenant boundary.

    Every row in every fact table carries `organization_id`, and it is the
    leading column of every composite index. Row-level security policies are
    defined against it so that a missing WHERE clause in application code
    cannot leak across tenants — defence in depth, because a cross-tenant cost
    leak in a FinOps product is an existential incident.
    """

    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    #: Negotiated discount applied across all rate cards for this tenant.
    discount_multiplier: Mapped[Decimal] = mapped_column(
        Numeric(6, 4), nullable=False, default=Decimal("1")
    )
    data_residency: Mapped[str | None] = mapped_column(String(32))
    settings: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    departments: Mapped[list[Department]] = relationship(back_populates="organization")


class Department(Base, TimestampMixin):
    __tablename__ = "departments"
    __table_args__ = (
        UniqueConstraint("organization_id", "code", name="uq_department_org_code"),
        Index("ix_departments_org", "organization_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    cost_center: Mapped[str | None] = mapped_column(String(64))
    #: Self-reference supports arbitrary org hierarchies (division -> department
    #: -> sub-department) without a separate closure table. Depth is small
    #: (< 5) so recursive CTEs are cheap.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("departments.id", ondelete="SET NULL")
    )
    owner_email: Mapped[str | None] = mapped_column(String(320))

    organization: Mapped[Organization] = relationship(back_populates="departments")
    teams: Mapped[list[Team]] = relationship(back_populates="department")


class Team(Base, TimestampMixin):
    __tablename__ = "teams"
    __table_args__ = (
        UniqueConstraint("organization_id", "slug", name="uq_team_org_slug"),
        Index("ix_teams_department", "department_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("departments.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(80), nullable=False)
    owner_email: Mapped[str | None] = mapped_column(String(320))

    department: Mapped[Department | None] = relationship(back_populates="teams")


class User(Base, TimestampMixin):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("organization_id", "email", name="uq_user_org_email"),
        Index("ix_users_org", "organization_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(200))
    #: Null when the tenant uses SSO — there is no local password to store,
    #: which is the desired end state for every enterprise deployment.
    hashed_password: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="viewer")
    team_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("teams.id", ondelete="SET NULL")
    )
    external_subject: Mapped[str | None] = mapped_column(String(255), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ApiKey(Base, TimestampMixin):
    """SDK credentials.

    Only a hash is stored — the plaintext is shown once at creation. `prefix`
    holds the first 8 visible characters so the UI can identify a key in a list
    without being able to reconstruct it.
    """

    __tablename__ = "api_keys"
    __table_args__ = (Index("ix_api_keys_org_active", "organization_id", "is_active"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    prefix: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    hashed_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    scopes: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


# ---------------------------------------------------------------------------
# Pricing catalog
# ---------------------------------------------------------------------------


class ModelCatalog(Base, TimestampMixin):
    __tablename__ = "model_catalog"
    __table_args__ = (
        UniqueConstraint("provider", "model", name="uq_model_catalog_provider_model"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    provider: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(160), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    context_window: Mapped[int] = mapped_column(Integer, nullable=False, default=128_000)
    max_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=16_384)
    quality_index: Mapped[Decimal] = mapped_column(
        Numeric(4, 3), nullable=False, default=Decimal("0.75")
    )
    latency_ms_per_1k_output: Mapped[int] = mapped_column(Integer, nullable=False, default=1000)
    supports_prompt_cache: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    supports_batch: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    supports_vision: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    supports_tools: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_deprecated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)


class ModelPricing(Base, TimestampMixin):
    """Effective-dated rate card. Never updated in place — see pricing.py."""

    __tablename__ = "model_pricing"
    __table_args__ = (
        Index("ix_model_pricing_lookup", "provider", "model", "effective_from"),
        CheckConstraint("effective_to IS NULL OR effective_to > effective_from",
                        name="ck_model_pricing_valid_range"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[str] = mapped_column(String(160), nullable=False)
    #: Per-1M-token rates keyed by token class, matching the provider's own
    #: published unit so a human can diff it against a pricing page.
    rates: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    per_request: Mapped[Decimal] = mapped_column(
        Numeric(20, 10), nullable=False, default=Decimal("0")
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    effective_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(String(40), nullable=False, default="provider_sync")


# ---------------------------------------------------------------------------
# Usage facts (partitioned)
# ---------------------------------------------------------------------------


class UsageEventRow(Base):
    """Raw metered request.

    Note the composite primary key `(id, occurred_at)`: Postgres requires the
    partition key to be part of every unique constraint on a partitioned table.
    This is a common migration trap — a plain `id` PK simply will not create.
    """

    __tablename__ = "usage_events"
    __table_args__ = (
        UniqueConstraint(
            "organization_id", "idempotency_key", "occurred_at",
            name="uq_usage_events_idempotency",
        ),
        Index("ix_usage_events_org_time", "organization_id", "occurred_at"),
        Index("ix_usage_events_model_time", "organization_id", "provider", "model", "occurred_at"),
        Index("ix_usage_events_team_time", "organization_id", "team_id", "occurred_at"),
        Index("ix_usage_events_feature_time", "organization_id", "feature", "occurred_at"),
        Index("ix_usage_events_conversation", "conversation_id",
              postgresql_where="conversation_id IS NOT NULL"),
        Index("ix_usage_events_agent_run", "agent_run_id",
              postgresql_where="agent_run_id IS NOT NULL"),
        Index("ix_usage_events_fingerprint", "organization_id", "prompt_fingerprint",
              postgresql_where="prompt_fingerprint IS NOT NULL"),
        {"postgresql_partition_by": "RANGE (occurred_at)"},
    )

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(128))

    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[str] = mapped_column(String(160), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False, default="chat")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="success")
    error_code: Mapped[str | None] = mapped_column(String(64))

    # Token counts. BigInteger because a single batch embedding call can exceed
    # the 2.1B int4 ceiling when summed in a rollup, and an overflow in a
    # billing column is not a recoverable error.
    input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    embedding_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    image_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    audio_input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    audio_output_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    # Attribution
    department_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    team_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    user_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    project: Mapped[str | None] = mapped_column(String(120))
    feature: Mapped[str | None] = mapped_column(String(120))
    application: Mapped[str | None] = mapped_column(String(120))
    environment: Mapped[str] = mapped_column(String(32), nullable=False, default="production")
    customer_id: Mapped[str | None] = mapped_column(String(120))
    cost_center: Mapped[str | None] = mapped_column(String(64))
    tags: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Trace
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    time_to_first_token_ms: Mapped[int | None] = mapped_column(Integer)
    streamed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    parent_request_id: Mapped[str | None] = mapped_column(String(64))
    agent_run_id: Mapped[str | None] = mapped_column(String(64))
    agent_step: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    conversation_id: Mapped[str | None] = mapped_column(String(64))
    conversation_turn: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rag_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rag_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    system_prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    few_shot_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_definition_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gpu_seconds: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False, default=Decimal("0"))
    network_gb: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False, default=Decimal("0"))

    prompt_template_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    prompt_version: Mapped[str | None] = mapped_column(String(40))
    #: Hash only — never prompt text. See prompt_optimizer.py on privacy.
    prompt_fingerprint: Mapped[str | None] = mapped_column(String(64))
    served_from_semantic_cache: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # Resolved cost, denormalised onto the event. See cost_engine.py for why
    # pricing happens at write time.
    total_cost: Mapped[Decimal] = mapped_column(Numeric(20, 10), nullable=False, default=Decimal("0"))
    token_cost: Mapped[Decimal] = mapped_column(Numeric(20, 10), nullable=False, default=Decimal("0"))
    infrastructure_cost: Mapped[Decimal] = mapped_column(
        Numeric(20, 10), nullable=False, default=Decimal("0")
    )
    cache_savings: Mapped[Decimal] = mapped_column(
        Numeric(20, 10), nullable=False, default=Decimal("0")
    )
    cost_by_token_class: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    rate_card_version: Mapped[str | None] = mapped_column(String(120))
    is_wasted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)


class UsageRollupDaily(Base):
    """Pre-aggregated daily facts. Every dashboard tile reads from here.

    The grain is deliberately wide — org, date, department, team, provider,
    model, feature, environment. Wider grain means more rows but avoids the
    trap of needing a second rollup table every time a new facet is added to
    the UI. At realistic cardinality this is ~10^5 rows/day/tenant, which is
    three orders of magnitude smaller than the raw events it replaces.
    """

    __tablename__ = "usage_rollup_daily"
    __table_args__ = (
        UniqueConstraint(
            "organization_id", "bucket_date", "department_id", "team_id",
            "provider", "model", "feature", "environment",
            name="uq_rollup_daily_grain",
        ),
        Index("ix_rollup_daily_org_date", "organization_id", "bucket_date"),
        Index("ix_rollup_daily_team", "organization_id", "team_id", "bucket_date"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    bucket_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    department_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    team_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[str] = mapped_column(String(160), nullable=False)
    feature: Mapped[str | None] = mapped_column(String(120))
    environment: Mapped[str] = mapped_column(String(32), nullable=False, default="production")

    request_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    retry_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    total_cost: Mapped[Decimal] = mapped_column(Numeric(20, 10), nullable=False, default=Decimal("0"))
    wasted_cost: Mapped[Decimal] = mapped_column(Numeric(20, 10), nullable=False, default=Decimal("0"))
    cache_savings: Mapped[Decimal] = mapped_column(
        Numeric(20, 10), nullable=False, default=Decimal("0")
    )
    latency_ms_sum: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    #: Stored so p95 can be reported without re-reading raw events. Computed
    #: from a t-digest sketch in the worker; the sketch itself lives in JSONB
    #: so rollups remain mergeable across time buckets.
    latency_sketch: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------


class BudgetRow(Base, TimestampMixin):
    __tablename__ = "budgets"
    __table_args__ = (
        Index("ix_budgets_org_scope", "organization_id", "scope", "scope_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(120), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    period: Mapped[str] = mapped_column(String(16), nullable=False, default="monthly")
    alert_thresholds: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    action_at_limit: Mapped[str] = mapped_column(String(32), nullable=False, default="warn")
    hard_stop_multiplier: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    rollover: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    owner_email: Mapped[str | None] = mapped_column(String(320))


class PolicyRow(Base, TimestampMixin):
    __tablename__ = "policies"
    __table_args__ = (Index("ix_policies_org_enabled", "organization_id", "is_enabled"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    scope: Mapped[str] = mapped_column(String(32), nullable=False, default="organization")
    scope_id: Mapped[str | None] = mapped_column(String(120))
    action: Mapped[str] = mapped_column(String(32), nullable=False, default="warn")
    rules: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    fail_closed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class RecommendationRow(Base, TimestampMixin):
    __tablename__ = "recommendations"
    __table_args__ = (
        UniqueConstraint("organization_id", "kind", "scope_key", name="uq_recommendation_dedupe"),
        Index("ix_recommendations_org_status", "organization_id", "status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[str] = mapped_column(String(48), nullable=False)
    scope_key: Mapped[str] = mapped_column(String(200), nullable=False)
    estimated_monthly_savings: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    priority_score: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False, default=Decimal("0"))
    effort: Mapped[str] = mapped_column(String(32), nullable=False, default="config_change")
    risk: Mapped[str] = mapped_column(String(16), nullable=False, default="low")
    quality_impact: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False, default=Decimal("0"))
    requires_evaluation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    blocked_by_quality: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="open")
    implementation_steps: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: Realised saving measured after the change was applied. The gap between
    #: this and `estimated_monthly_savings` is the platform's own accuracy
    #: record, reported openly rather than quietly discarded.
    realised_monthly_savings: Mapped[Decimal | None] = mapped_column(Numeric(20, 4))
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    applied_by: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))


class AnomalyRow(Base, TimestampMixin):
    __tablename__ = "anomalies"
    __table_args__ = (
        Index("ix_anomalies_org_detected", "organization_id", "detected_at"),
        Index("ix_anomalies_open", "organization_id", "is_resolved"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[str] = mapped_column(String(48), nullable=False)
    scope_key: Mapped[str | None] = mapped_column(String(200))
    observed_value: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False, default=Decimal("0"))
    expected_value: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False, default=Decimal("0"))
    deviation_score: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False, default=Decimal("0"))
    estimated_impact: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False, default=Decimal("0"))
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    is_resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_note: Mapped[str | None] = mapped_column(Text)


class ForecastRow(Base, TimestampMixin):
    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint("organization_id", "scope", "scope_id", "generated_for",
                         name="uq_forecast_scope_date"),
        Index("ix_forecasts_org_scope", "organization_id", "scope", "scope_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(120), nullable=False)
    generated_for: Mapped[datetime] = mapped_column(Date, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    method: Mapped[str] = mapped_column(String(48), nullable=False, default="holt_winters")
    points: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    mape: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    confidence: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False, default=Decimal("0.8"))
    warnings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)


class PromptTemplate(Base, TimestampMixin):
    __tablename__ = "prompt_templates"
    __table_args__ = (
        UniqueConstraint("organization_id", "slug", name="uq_prompt_template_slug"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(120), nullable=False)
    owner_team_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    feature: Mapped[str | None] = mapped_column(String(120))
    active_version: Mapped[str | None] = mapped_column(String(40))
    description: Mapped[str | None] = mapped_column(Text)


class PromptVersion(Base, TimestampMixin):
    """A version of a prompt template.

    `body` is nullable by design: tenants operating under strict data policies
    keep prompt text entirely in their own systems and register only the
    fingerprint and derived metrics here. The platform is fully functional in
    that mode — only the in-app prompt studio editor is unavailable.
    """

    __tablename__ = "prompt_versions"
    __table_args__ = (
        UniqueConstraint("template_id", "version", name="uq_prompt_version"),
        Index("ix_prompt_versions_fingerprint", "fingerprint"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    template_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("prompt_templates.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[str] = mapped_column(String(40), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    body: Mapped[str | None] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    static_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    efficiency_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 2), nullable=False, default=Decimal("0")
    )
    analysis: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class QualityScore(Base, TimestampMixin):
    __tablename__ = "quality_scores"
    __table_args__ = (
        Index("ix_quality_subject", "organization_id", "subject", "dimension", "measured_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    subject: Mapped[str] = mapped_column(String(200), nullable=False)
    dimension: Mapped[str] = mapped_column(String(40), nullable=False)
    score: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False)
    stddev: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False, default=Decimal("0"))
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="deterministic")
    measured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)


class AuditLog(Base):
    """Append-only audit trail.

    No `updated_at`, no UPDATE grants for the application role — the table is
    INSERT-and-SELECT only at the database privilege level, so tampering
    requires a separate credential. SOC 2 CC7.2 and ISO 27001 A.12.4 both
    require this property to be enforced technically, not merely by convention.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_org_time", "organization_id", "occurred_at"),
        Index("ix_audit_actor", "organization_id", "actor_id", "occurred_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    actor_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    actor_email: Mapped[str | None] = mapped_column(String(320))
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(120))
    #: Before/after snapshot for mutations, so a reviewer can reconstruct state
    #: without replaying the whole log.
    changes: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(400))
    request_id: Mapped[str | None] = mapped_column(String(64))


class ApprovalRequestRow(Base, TimestampMixin):
    __tablename__ = "approval_requests"
    __table_args__ = (
        Index("ix_approvals_org_status", "organization_id", "status"),
        CheckConstraint("requester_id <> approver_id OR approver_id IS NULL",
                        name="ck_approval_segregation_of_duties"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    organization_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    requester_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    approver_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    subject: Mapped[str] = mapped_column(String(300), nullable=False)
    estimated_cost: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False)
    justification: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending")
    decision_note: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
