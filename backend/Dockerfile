# Stage 1: Builder
FROM rust:1.82-bookworm AS builder

RUN apt-get update && apt-get install -y \
    cmake \
    clang \
    llvm \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/

RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/webraw-backend /app/webraw-backend

RUN mkdir -p /app/cache

ENV LISTEN_ADDR=0.0.0.0:8080
ENV CACHE_DIR=/app/cache
ENV LOG_LEVEL=info

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

CMD ["/app/webraw-backend"]
