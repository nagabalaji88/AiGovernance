"""FastAPI dependencies: authentication, tenancy and store resolution."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings
from app.core.security import AuthError, PermissionDenied, Principal, decode_token
from app.domain.enums import Role
from app.store import DEMO_ORG_ID, DEMO_USER_ID, AnalyticsStore, default_store

# auto_error=False so we can distinguish "no credentials" from "bad
# credentials" and return a useful message rather than a bare 403.
_bearer = HTTPBearer(auto_error=False)


def get_store() -> AnalyticsStore:
    """Resolve the analytics store.

    Overridden in production wiring to return the Postgres-backed repository,
    and in tests to return an isolated instance per test.
    """
    return default_store


async def get_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> Principal:
    """Resolve the calling principal from a bearer token or API key.

    In non-production environments with no credentials supplied, a demo
    principal is returned so the API is explorable immediately after
    `make dev`. That fallback is gated on `settings.is_production` and refuses
    to activate in production — an unauthenticated path that silently grants
    owner rights would be the single worst defect this codebase could ship, so
    the guard is explicit rather than implied by configuration.
    """
    if credentials and credentials.credentials:
        try:
            claims = decode_token(credentials.credentials, expected_type="access")
        except AuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        from uuid import UUID

        return Principal(
            organization_id=UUID(claims["org"]),
            user_id=UUID(claims["sub"]),
            role=Role(claims.get("role", "viewer")),
            jti=claims.get("jti"),
        )

    if x_api_key:
        # The production implementation looks the key up by its indexed prefix
        # and bcrypt-verifies the remainder; see core/security.py. Until the
        # repository layer is wired, an API key is accepted in non-production
        # only, with ingestion scopes.
        if settings.is_production:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key authentication requires the key store",
            )
        return Principal(
            organization_id=DEMO_ORG_ID,
            is_service=True,
            scopes=["usage:write", "usage:read"],
        )

    if settings.is_production:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return Principal(
        organization_id=DEMO_ORG_ID,
        user_id=DEMO_USER_ID,
        role=Role.OWNER,
        email="demo@localhost",
    )


def require(permission: str):  # type: ignore[no-untyped-def]
    """Route-level permission guard.

    Usage: `dependencies=[Depends(require("budget:write"))]`. Handlers that
    need the principal anyway call `principal.require(...)` directly instead,
    which avoids resolving it twice.
    """

    async def _guard(principal: Principal = Depends(get_principal)) -> Principal:
        try:
            principal.require(permission)
        except PermissionDenied as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        return principal

    return _guard
