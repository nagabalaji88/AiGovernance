"""Canonical enumerations shared across the domain.

These values are persisted in the database and appear in the public API, so
they are treated as a compatibility surface: add new members freely, never
rename or remove an existing one without a migration + API version bump.
"""

from __future__ import annotations

from enum import StrEnum


class Provider(StrEnum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    GOOGLE_GEMINI = "google_gemini"
    AWS_BEDROCK = "aws_bedrock"
    MISTRAL = "mistral"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    OLLAMA = "ollama"
    VLLM = "vllm"
    HUGGINGFACE = "huggingface"
    CUSTOM_REST = "custom_rest"


#: Providers whose cost is dominated by owned infrastructure (GPU-seconds,
#: energy, amortised hardware) rather than a per-token vendor invoice.
SELF_HOSTED_PROVIDERS: frozenset[Provider] = frozenset(
    {Provider.OLLAMA, Provider.VLLM, Provider.HUGGINGFACE, Provider.CUSTOM_REST}
)


class ModelType(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    REASONING = "reasoning"
    VISION = "vision"
    SPEECH = "speech"
    IMAGE = "image"
    MULTIMODAL = "multimodal"
    LOCAL = "local"


class TokenClass(StrEnum):
    """Billable token categories.

    Providers meter these at different rates; collapsing them into a single
    "tokens" number is the single largest source of cost-attribution error we
    are trying to eliminate.
    """

    INPUT = "input"
    OUTPUT = "output"
    CACHED_INPUT = "cached_input"
    CACHE_WRITE = "cache_write"
    REASONING = "reasoning"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO_INPUT = "audio_input"
    AUDIO_OUTPUT = "audio_output"


class RequestStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    CANCELLED = "cancelled"
    FILTERED = "filtered"


#: Statuses that still incur provider charges. A timed-out or filtered request
#: is usually billed for whatever was generated before the cut-off.
BILLABLE_STATUSES: frozenset[RequestStatus] = frozenset(
    {RequestStatus.SUCCESS, RequestStatus.TIMEOUT, RequestStatus.FILTERED, RequestStatus.CANCELLED}
)


class RoutingObjective(StrEnum):
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    HIGHEST_QUALITY = "highest_quality"
    BALANCED = "balanced"
    REASONING = "reasoning"
    VISION = "vision"
    EMBEDDING = "embedding"
    LOCAL_ONLY = "local_only"


class TaskComplexity(StrEnum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class RecommendationKind(StrEnum):
    MODEL_DOWNGRADE = "model_downgrade"
    PROVIDER_SWITCH = "provider_switch"
    PROMPT_COMPRESSION = "prompt_compression"
    CONTEXT_REDUCTION = "context_reduction"
    SEMANTIC_CACHE = "semantic_cache"
    PROMPT_CACHE = "prompt_cache"
    EMBEDDING_MODEL_SWAP = "embedding_model_swap"
    RAG_TOPK_REDUCTION = "rag_topk_reduction"
    RAG_CHUNK_TUNING = "rag_chunk_tuning"
    HISTORY_SUMMARISATION = "history_summarisation"
    REQUEST_BATCHING = "request_batching"
    RETRY_POLICY = "retry_policy"
    DEDUPLICATION = "deduplication"


class RecommendationStatus(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    APPLIED = "applied"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


class AnomalyKind(StrEnum):
    COST_SPIKE = "cost_spike"
    TOKEN_SPIKE = "token_spike"
    LATENCY_SPIKE = "latency_spike"
    ERROR_BURST = "error_burst"
    RETRY_STORM = "retry_storm"
    RUNAWAY_AGENT = "runaway_agent"
    INFINITE_LOOP = "infinite_loop"
    CONTEXT_EXPLOSION = "context_explosion"
    PROVIDER_DRIFT = "provider_drift"
    PROMPT_INJECTION = "prompt_injection"
    API_ABUSE = "api_abuse"
    RAG_MISCONFIGURATION = "rag_misconfiguration"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BudgetPeriod(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class BudgetScope(StrEnum):
    ORGANIZATION = "organization"
    DEPARTMENT = "department"
    TEAM = "team"
    USER = "user"
    PROJECT = "project"
    FEATURE = "feature"
    MODEL = "model"
    PROVIDER = "provider"


class EnforcementAction(StrEnum):
    """What the policy engine does when a budget or policy threshold trips."""

    ALLOW = "allow"
    WARN = "warn"
    REQUIRE_APPROVAL = "require_approval"
    DOWNGRADE_MODEL = "downgrade_model"
    THROTTLE = "throttle"
    BLOCK = "block"


class Role(StrEnum):
    """RBAC roles, ordered from least to most privileged within a tenant."""

    VIEWER = "viewer"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    FINOPS = "finops"
    APPROVER = "approver"
    ADMIN = "admin"
    OWNER = "owner"


class QualityDimension(StrEnum):
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    GROUNDEDNESS = "groundedness"
    RELEVANCE = "relevance"
    CITATION_QUALITY = "citation_quality"
    HALLUCINATION_RATE = "hallucination_rate"
    USER_SATISFACTION = "user_satisfaction"
    TASK_SUCCESS = "task_success"
