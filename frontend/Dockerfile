# ---- builder ---------------------------------------------------------------
FROM node:22-alpine AS builder

WORKDIR /build

# Copy manifests first so the dependency layer is cached independently of
# source changes — a source-only edit then rebuilds in seconds rather than
# re-resolving the whole tree.
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY . .
RUN npm run build

# ---- runtime ---------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

# nginx's default config runs the master as root to bind port 80. We serve on
# 8080 instead so the whole process tree can run unprivileged — binding below
# 1024 is the only reason that image needs root, and the load balancer does not
# care which port the pod listens on.
RUN adduser -D -u 10001 app \
    && mkdir -p /var/cache/nginx /var/run \
    && chown -R app:app /var/cache/nginx /var/run /usr/share/nginx/html

COPY --from=builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -qO- http://localhost:8080/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
