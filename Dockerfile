# Build stage
FROM ubuntu:22.04 AS builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    ninja-build \
    libnats-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source
WORKDIR /app
COPY . .

# Build CPU version
RUN mkdir build && cd build && \
    cmake .. -DSOVALUNE_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=Release -G Ninja && \
    ninja

# Runtime stage
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libnats-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /app/build/sovalune-model-runtime /usr/local/bin/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run
ENTRYPOINT ["sovalune-model-runtime"]
CMD ["--help"]
