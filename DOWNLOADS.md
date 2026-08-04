# Downloads

## AiGovernance Python 3.9-Compatible Release

**File:** `AiGovernance-python39-ready.zip` (1.5 MB)

A complete, ready-to-deploy version of the AI Cost Intelligence & Token Optimization Platform with full Python 3.9 compatibility.

### What's Included

- Complete backend (Python 3.9–3.13 compatible)
- Complete frontend (React with Vite)
- All documentation (architecture, API, security, operations, roadmap)
- Kubernetes manifests and infrastructure configs
- 176 comprehensive tests
- CI/CD pipeline configuration

### What's Excluded

- `.git` directory (start fresh with `git init` or clone the repo)
- Build artifacts (`__pycache__`, `.pytest_cache`, `dist`, `build`)
- Node and pip caches
- Credentials and `.env` files
- Large compiled assets (use `make install` to rebuild)

### Python 3.9 Compatibility Changes

This release includes all changes necessary to run on Python 3.9:

- **Type annotations:** All `X | None` unions converted to `Optional[X]`
- **StrEnum backport:** Custom `StrEnum` class for 3.9 (byte-identical to Python 3.11 native)
- **datetime.UTC:** Replaced with `timezone.utc` for 3.9 compatibility
- **Dataclass slots:** Removed `slots=True` parameter (unsupported in 3.9)
- **Itertools:** Removed `itertools.pairwise` and `zip(strict=...)` calls
- **Dependencies:** Updated to versions supporting 3.9 (numpy <2.1, starlette compatible)
- **CI matrix:** Tests both Python 3.9 and 3.13 to prevent regressions

### Quick Start

1. **Extract the archive:**
   ```bash
   unzip AiGovernance-python39-ready.zip
   cd AiGovernance
   ```

2. **Install dependencies:**
   ```bash
   make install
   ```

3. **Run demo with synthetic data:**
   ```bash
   make demo
   ```
   - Dashboards: http://127.0.0.1:5173
   - API docs: http://127.0.0.1:8000/docs

4. **Run full production topology (Postgres, Redis, Celery, Prometheus, Grafana):**
   ```bash
   make up
   ```

5. **Run tests:**
   ```bash
   make test       # 176 tests with coverage
   make check      # all CI checks (lint, types, tests, security, bundle size)
   ```

### System Requirements

- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- Node.js 22
- Docker & Docker Compose (for `make up`)
- Make
- ~500 MB disk space (excluding node_modules)

### Security Notes

Python 3.9 reached end-of-life in October 2025. While this release maintains compatibility, note that:

- The project has 13 advisories on Python 3.9 vs 1 on Python 3.11
- 12 of these are unfixable (patched versions require ≥3.10)
- Key advisories: starlette (x5), orjson, click, python-dotenv
- See `docs/ROADMAP.md` for full security analysis

**Recommendation:** Use Python 3.11+ for production deployments when possible. The 3.9 floor exists solely for deployment environments that require it.

### Documentation

- **[README.md](README.md)** — Product overview
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — System design, data model, scaling
- **[docs/API.md](docs/API.md)** — API endpoint reference
- **[docs/SECURITY.md](docs/SECURITY.md)** — Threat model, RBAC, compliance
- **[docs/OPERATIONS.md](docs/OPERATIONS.md)** — Deployment, SLOs, runbooks
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — Sprint plan, risk register, future work

### Development

Clone the repository instead for an editable checkout:

```bash
git clone https://github.com/nagabalaji88/AiGovernance.git
cd AiGovernance
git checkout claude/ai-cost-intelligence-platform-eo7fp6
make install
```

### Support

For issues, questions, or contributions, see the repository's issue tracker or contact the development team.

---

**Last Updated:** 2026-08-04  
**Python Support:** 3.9–3.13  
**Node Support:** 22+
