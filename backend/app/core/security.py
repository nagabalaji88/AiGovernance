"""Authentication and authorization primitives.

## Token strategy

Short-lived access JWTs (30 min) plus long-lived refresh tokens. The access
token carries `org_id` and `role` as claims so authorization needs no database
round trip on the hot path — important because every dashboard tile is a
separate request and a per-request user lookup would dominate p95.

The cost of stateless tokens is revocation: a role downgrade does not take
effect until the access token expires. Mitigated by keeping the TTL short and
maintaining a Redis deny-list keyed on `jti` for immediate revocation on
logout, password change, or user deactivation. Checking the deny-list is one
Redis GET, which is affordable; querying Postgres for the user's current role
on every request is not.

## API keys vs. JWTs

SDK ingestion authenticates with API keys, not JWTs — machine clients cannot
perform an interactive refresh, and a long-lived JWT is strictly worse than an
API key that can be scoped and revoked individually. Keys are stored as bcrypt
hashes; the plaintext exists only at creation time.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings
from app.domain.enums import Role
from app.services.governance import has_permission

#: bcrypt with the default cost factor of 12: ~250ms per hash on current
#: hardware, which is the intended cost. Faster schemes (or a lower cost) make
#: offline cracking of a leaked hash dump cheap.
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

API_KEY_PREFIX = "aick_"


class AuthError(Exception):
    """Authentication failure. Mapped to 401 by the exception handler."""


class PermissionDenied(Exception):
    """Authorization failure. Mapped to 403."""


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def create_access_token(
    *,
    subject: str,
    organization_id: str,
    role: Role | str,
    extra_claims: dict[str, Any] | None = None,
    expires_delta: timedelta | None = None,
) -> tuple[str, str]:
    """Return `(token, jti)`. The jti is retained for revocation."""
    settings = get_settings()
    now = datetime.now(UTC)
    expiry = now + (expires_delta or timedelta(minutes=settings.access_token_ttl_minutes))
    jti = secrets.token_urlsafe(16)
    claims: dict[str, Any] = {
        "sub": subject,
        "org": organization_id,
        "role": str(role),
        "iat": int(now.timestamp()),
        "exp": int(expiry.timestamp()),
        "jti": jti,
        "typ": "access",
        **(extra_claims or {}),
    }
    token = jwt.encode(claims, settings.secret_key, algorithm=settings.jwt_algorithm)
    return token, jti


def create_refresh_token(*, subject: str, organization_id: str) -> tuple[str, str]:
    settings = get_settings()
    now = datetime.now(UTC)
    expiry = now + timedelta(days=settings.refresh_token_ttl_days)
    jti = secrets.token_urlsafe(24)
    claims = {
        "sub": subject,
        "org": organization_id,
        "iat": int(now.timestamp()),
        "exp": int(expiry.timestamp()),
        "jti": jti,
        "typ": "refresh",
    }
    return jwt.encode(claims, settings.secret_key, algorithm=settings.jwt_algorithm), jti


def decode_token(token: str, *, expected_type: str = "access") -> dict[str, Any]:
    """Decode and validate a JWT.

    The `typ` check is not optional: without it a refresh token — which is
    long-lived by design — would be accepted as an access token, silently
    turning a 30-minute credential into a 7-day one.
    """
    settings = get_settings()
    try:
        claims = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise AuthError("invalid or expired token") from exc
    if claims.get("typ") != expected_type:
        raise AuthError(f"expected a {expected_type} token")
    return claims


def generate_api_key() -> tuple[str, str, str]:
    """Return `(plaintext, prefix, hashed)`.

    The plaintext is returned exactly once and never persisted. The prefix is
    stored separately so the key can be identified in a list view and looked up
    cheaply — bcrypt cannot be queried, so without an indexed prefix, verifying
    a key would mean bcrypt-comparing against every key in the tenant.
    """
    raw = secrets.token_urlsafe(32)
    plaintext = f"{API_KEY_PREFIX}{raw}"
    prefix = plaintext[: len(API_KEY_PREFIX) + 8]
    return plaintext, prefix, _pwd_context.hash(plaintext)


def verify_api_key(plaintext: str, hashed: str) -> bool:
    try:
        return _pwd_context.verify(plaintext, hashed)
    except ValueError:
        return False


def api_key_prefix(plaintext: str) -> str:
    return plaintext[: len(API_KEY_PREFIX) + 8]


def constant_time_compare(a: str, b: str) -> bool:
    """Timing-safe comparison for webhook signatures and similar secrets."""
    return hmac.compare_digest(a.encode(), b.encode())


def sign_payload(payload: bytes, secret: str) -> str:
    """HMAC-SHA256 signature for outbound webhooks, so receivers can verify us."""
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


class Principal:
    """The authenticated caller, resolved once per request."""

    __slots__ = ("email", "is_service", "jti", "organization_id", "role", "scopes", "user_id")

    def __init__(
        self,
        *,
        organization_id: UUID,
        user_id: UUID | None = None,
        role: Role = Role.VIEWER,
        email: str | None = None,
        is_service: bool = False,
        scopes: list[str] | None = None,
        jti: str | None = None,
    ) -> None:
        self.organization_id = organization_id
        self.user_id = user_id
        self.role = role
        self.email = email
        self.is_service = is_service
        self.scopes = scopes or []
        self.jti = jti

    def require(self, permission: str) -> None:
        """Raise unless the principal holds `permission`.

        Service principals (API keys) are checked against their explicit scope
        list rather than a role. A key minted for ingestion must not be usable
        to read the whole organization's cost data — that is the difference
        between a leaked ingestion key being an annoyance and being a breach.
        """
        if self.is_service:
            if "*" in self.scopes or permission in self.scopes:
                return
            raise PermissionDenied(f"api key lacks scope '{permission}'")
        if not has_permission(self.role, permission):
            raise PermissionDenied(f"role '{self.role}' lacks permission '{permission}'")

    def can(self, permission: str) -> bool:
        try:
            self.require(permission)
        except PermissionDenied:
            return False
        return True
