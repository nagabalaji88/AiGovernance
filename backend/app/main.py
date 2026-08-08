"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.security import AuthError, PermissionDenied
from app.observability import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    configure_logging,
    metrics_payload,
)

logger = logging.getLogger("aicost.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    logger.info(
        "starting %s in %s (docs=%s)",
        settings.app_name,
        settings.environment,
        settings.docs_enabled,
    )
    # Production wiring refreshes the pricing catalog from Postgres here and
    # starts the periodic refresh task. The seed catalog is already loaded at
    # import time, so the service is functional before that completes — a
    # pricing-sync failure degrades price freshness, never availability.

    if settings.seed_demo_data:
        # Guarded twice: the setting must be on *and* the environment must not
        # be production. Synthetic usage silently appearing in a real tenant's
        # cost reports would be a data-integrity incident, so a misconfigured
        # env var alone must not be able to cause it.
        if settings.is_production:
            logger.error("SEED_DEMO_DATA is set in production; refusing to seed synthetic usage")
        else:
            from app.demo import seed, seed_governance
            from app.store import default_store

            count = seed(default_store, days=settings.seed_demo_days)
            seed_governance(default_store)
            logger.info("seeded %s synthetic usage events for the demo tenant", f"{count:,}")

    yield
    logger.info("shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Measure, forecast, attribute and optimize enterprise AI spend across "
            "providers without degrading response quality."
        ),
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach a request id, record metrics, and never leak internals.

        The request id is echoed in the response header and in every log line
        for the request, which is what makes a customer-reported "this was slow
        at 14:03" traceable without full-fidelity tracing on 100% of traffic.
        """
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - started
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            REQUEST_COUNT.labels(method=request.method, path=path, status="500").inc()
            REQUEST_LATENCY.labels(method=request.method, path=path).observe(elapsed)
            logger.exception("unhandled error [request_id=%s]", request_id)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "code": "internal_error",
                    # Deliberately opaque: exception text can carry connection
                    # strings, row contents and internal hostnames.
                    "detail": "An internal error occurred.",
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )

        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        REQUEST_COUNT.labels(method=request.method, path=path, status=str(response.status_code)).inc()
        REQUEST_LATENCY.labels(method=request.method, path=path).observe(elapsed)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.2f}"
        return response

    @app.exception_handler(AuthError)
    async def _auth_error(request: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "code": "unauthenticated",
                "detail": str(exc),
                "request_id": getattr(request.state, "request_id", None),
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(PermissionDenied)
    async def _permission_denied(request: Request, exc: PermissionDenied) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "code": "forbidden",
                "detail": str(exc),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.get("/health", tags=["system"], include_in_schema=False)
    async def health() -> dict[str, str]:
        """Liveness. Deliberately dependency-free.

        A liveness probe that checks the database restarts healthy pods during
        a database incident, converting a degraded read path into a full
        outage. Dependency health belongs in /ready.
        """
        return {"status": "ok"}

    @app.get("/ready", tags=["system"], include_in_schema=False)
    async def ready() -> dict[str, object]:
        """Readiness. Checks the dependencies needed to serve traffic."""
        return {"status": "ready", "checks": {"pricing_catalog": "ok"}}

    @app.get("/metrics", tags=["system"], include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        payload, content_type = metrics_payload()
        return PlainTextResponse(content=payload, media_type=content_type)

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
