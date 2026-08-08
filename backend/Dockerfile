# Multi-stage build.
#
# Stage separation exists to keep build tooling (compilers, headers, pip cache)
# out of the runtime image. The result is both smaller and materially safer:
# an attacker who achieves execution in a container with no compiler and no
# package manager has far fewer options than one in a full build environment.

# ---- builder ---------------------------------------------------------------
FROM python:3.9-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is needed for any dependency without a manylinux wheel; it
# never reaches the runtime stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml ./
COPY app ./app

# Install into an isolated prefix that is copied wholesale into the runtime
# image, so the runtime carries dependencies without a package manager.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && pip install .

# ---- runtime ---------------------------------------------------------------
FROM python:3.9-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Non-root by default. The Kubernetes manifests also enforce
# runAsNonRoot, but a container that only refuses to run as root when the
# orchestrator says so is not hardened — it is hardened by its scheduler.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

# curl for the container healthcheck only; no compilers, no build tooling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
WORKDIR /srv
COPY --chown=app:app app ./app
COPY --chown=app:app alembic.ini ./alembic.ini
COPY --chown=app:app alembic ./alembic

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Workers are set via env so the same image serves both the API deployment and
# the single-worker ingestion pods. Uvicorn rather than Gunicorn: the workload
# is almost entirely async I/O, where Gunicorn's process manager adds a layer
# without adding throughput.
ENV WEB_CONCURRENCY=4
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY} --no-access-log"]
