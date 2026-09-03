# Sovalune Model Runtime

C++/CUDA inference engine for Sovalune AI models.

## Overview

This component handles:
- Online inference with quantized models (GGUF format)
- Tool calling loop (function calling)
- Streaming token generation
- CPU and CUDA backends

## Building

### Prerequisites

- CMake 3.20+
- C++20 compiler
- libnats-dev

### CPU Only (Development)

```bash
mkdir build && cd build
cmake .. -DSOVALUNE_CPU_ONLY=ON
make -j$(nproc)
```

### With CUDA

```bash
mkdir build-cuda && cd build-cuda
cmake .. -DSOVALUNE_USE_CUDA=ON -DSOVALUNE_CPU_ONLY=OFF
make -j$(nproc)
```

## Running

```bash
export SOVALUNE_MODEL_PATH="/path/to/model.gguf"
export SOVALUNE_RUNTIME_MODE="cpu"
export SOVALUNE_NATS_URL="nats://localhost:4222"
./build/sovalune-model-runtime
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SOVALUNE_MODEL_PATH` | | Path to GGUF model file |
| `SOVALUNE_RUNTIME_MODE` | `cpu` | `cpu` or `cuda` |
| `SOVALUNE_NATS_URL` | `nats://localhost:4222` | NATS connection URL |

## Architecture

```
src/
├── engine/           # Inference engine wrapper
├── tool_calling/     # Tool call parsing and execution
├── nats_bridge/      # NATS client for communication
├── tokenizer/        # Token counting
└── main.cpp          # Entry point
```
