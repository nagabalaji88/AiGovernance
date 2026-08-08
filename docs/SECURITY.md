# Security

## Threat model

A FinOps platform for AI holds something unusual: a complete, structured map of
how an enterprise uses AI — which teams, which models, which features, at what
volume, growing how fast. That is competitive intelligence even without a
single prompt in it. Two consequences shape the whole design:

1. **The platform is a high-value target despite holding no prompts.** Metadata
   alone is sensitive. Egress is default-deny and the cloud metadata endpoint
   is explicitly blocked, because SSRF-to-instance-role is the standard path
   from "read access to a dashboard" to "read access to the account".

2. **Cross-tenant leakage is existential, not merely bad.** A bug that shows
   one customer another's spend ends the product. Isolation is therefore
   enforced at the database (row-level security keyed on `organization_id`),
   not only by application `WHERE` clauses — a forgotten filter must fail
   closed.

### Assets, ranked

| Asset | Impact if disclosed | Control |
|---|---|---|
| Cross-tenant usage data | Existential | RLS + `organization_id` leading every index; tenancy resolved from the token, never from a request parameter |
| Provider API keys | Direct financial loss | Never stored. The platform receives metered *usage*, never credentials |
| Prompt content | Customer PII exposure | Never ingested. Only fingerprints (SHA-256) and token counts |
| JWT signing key | Full impersonation | External secret store, never in image or ConfigMap; startup refuses the dev default in production |
| Audit log | Undetectable tampering | Append-only at the privilege level — no UPDATE grant for the app role |

## Authentication

**Humans** authenticate via OIDC/SAML against the enterprise IdP. When
`OIDC_ISSUER` is configured, local password auth is disabled outright — mixed
auth modes are a standing audit finding and a reliable source of orphaned
accounts.

**Machines** (SDKs) authenticate with API keys, not JWTs. A machine client
cannot perform an interactive refresh, and a long-lived JWT is strictly worse
than a scoped, individually revocable key. Keys are stored as bcrypt hashes
(cost 12); the plaintext exists once, at creation.

### Token design

Short-lived access JWTs (30 min) carry `org_id` and `role` as claims, so
authorization needs no database round trip — necessary because every dashboard
tile is a separate request.

The cost is revocation latency: a role downgrade does not take effect until the
token expires. Mitigated by a Redis deny-list keyed on `jti`, checked on every
request (one GET, affordable) rather than re-reading the user's role from
Postgres (a join per tile, not affordable).

Refresh tokens are validated with an explicit `typ` check. Without it, a
7-day refresh token would be accepted as a 30-minute access token — a silent
14x extension of every credential's lifetime.

## Authorization

Seven roles with an explicit, flat permission matrix (`ROLE_PERMISSIONS` in
`services/governance.py`). Flat rather than hierarchical because an auditor
asks "what can a FinOps analyst do" and a matrix answers it at a glance;
inheritance chains require reconstruction. Hierarchy is applied at *assignment*
time instead.

| Role | Notable grants | Notable denials |
|---|---|---|
| Viewer | Dashboards, usage, forecast | Everything else |
| Developer | Prompt read/write, simulation | Chargeback — cross-department cost is not developer-visible by default |
| Analyst | Report export, anomalies | Budget mutation |
| FinOps | Budgets, chargeback, recommendations | User management |
| Approver | Approval decisions | Policy authoring — the approver must not be able to weaken what they approve against |
| Admin | Everything except ownership transfer | — |
| Owner | All | — |

**Service principals are scoped, not roled.** An API key minted for ingestion
carries `usage:write` only. A leaked ingestion key is then an annoyance, not a
breach of the organization's entire cost picture.

**Segregation of duties** is enforced in the domain object, not the HTTP layer:
`ApprovalRequest.decide()` raises if requester and approver match, and a
`CHECK` constraint backs it at the database. An auditor testing the control
directly against the data model finds it holds.

## Data protection

| Layer | Control |
|---|---|
| In transit | TLS 1.3, HSTS with a one-year max-age, redirect enforced at ingress |
| At rest | Postgres transparent encryption; per-tenant KMS keys where residency requires it |
| Application | No prompt text, no completions, no provider credentials ever persisted |
| Backups | Encrypted, 35-day PITR, restore rehearsed quarterly |

### Why prompt text is never stored

`services/prompt_optimizer.py` accepts prompt text and returns only counts,
character offsets and hashes. Analysis runs either in the tenant's own process
(SDK) or against text explicitly submitted to the prompt studio, and the
persisted result carries no content.

This is a deliberate architectural boundary. It keeps the platform outside the
customer's PII, GDPR and data-residency scope entirely, which converts a
months-long privacy review into a short one. The cost is that semantic
paraphrase detection is weaker than an embedding-based approach would be — a
limitation documented in the module rather than quietly traded away.

A regression test asserts the invariant directly: analysing a prompt containing
a credit card number and asserting the serialised result does not contain it.

## Network

Default-deny egress (`infra/k8s/base.yaml`). Permitted destinations are
enumerated: Postgres, Redis, the OTel collector, DNS, and outbound 443 for
provider pricing sync. RFC1918 ranges and `169.254.169.254` are excluded from
the internet rule specifically to block SSRF pivots into the metadata service
and lateral movement into the VPC.

Ingress is restricted to the ingress controller, the frontend pods, and the
observability namespace.

## Container hardening

Every workload runs non-root, with a read-only root filesystem, all
capabilities dropped, `allowPrivilegeEscalation: false`, and the
`RuntimeDefault` seccomp profile. The namespace enforces the `restricted` Pod
Security Standard, so a manifest that omits these is rejected at admission
rather than silently running privileged.

Build tooling never reaches the runtime image (multi-stage build). A container
with no compiler and no package manager is a materially worse foothold.

The frontend serves on 8080 rather than 80 so nginx needs no root at all —
binding below 1024 is the only reason that image would.

## Compliance mapping

| Control | SOC 2 | ISO 27001 | GDPR | Implementation |
|---|---|---|---|---|
| Access control | CC6.1 | A.9.2 | Art. 32 | OIDC + RBAC + scoped keys |
| Audit trail | CC7.2 | A.12.4 | Art. 30 | Append-only `audit_logs`, no UPDATE grant |
| Encryption | CC6.7 | A.10.1 | Art. 32 | TLS 1.3, encryption at rest |
| Change management | CC8.1 | A.12.1 | — | PR review, CI gates, signed images |
| Segregation of duties | CC6.3 | A.6.1 | — | Approver ≠ requester, enforced in domain + DB |
| Data minimisation | — | A.8.2 | Art. 5(1)(c) | No prompt content ingested |
| Right to erasure | — | — | Art. 17 | Per-tenant deletion; partitions make it bounded work |
| Availability | A1.2 | A.17.1 | — | Multi-AZ, PDB, HPA, fail-open governance |

**HIPAA** is achievable but not claimed by default: it requires the self-hosted
deployment profile (`require_self_hosted` routing constraint), a BAA with every
retained provider, and audit retention extended to six years.

## Known limitations

Stated plainly, because a security document that lists only strengths is not
useful:

- **Revocation is not instant.** A role change takes effect within the access
  token TTL (30 min) unless the session is explicitly revoked via the deny-list.
- **Fail-open governance is a deliberate availability trade.** A policy-service
  outage means requests that should have been blocked are allowed. Tenants
  needing fail-closed set it per policy, and the choice is surfaced in the UI
  rather than buried in configuration.
- **Budget counters are eventually consistent.** Enforcement can overshoot a
  limit by the counter lag (seconds of spend). Documented in
  `services/governance.py`; the alternative is a database round trip on the
  inference hot path.
- **Semantic caching can return a wrong answer.** This is why it defaults to a
  0.97 similarity threshold, requires per-prompt opt-in, and is recommended
  with a shadow-run instruction rather than enabled automatically.
