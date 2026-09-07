# Sovalune Model Runtime

C++/CUDA inference engine for Sovalune AI models.

## Overview

High-performance inference runtime with support for:
- GGUF model format via llama.cpp
- CUDA acceleration with cuBLAS
- CPU-only mode for development
- REST API (OpenAI-compatible)
- NATS messaging for distributed inference
- Tool calling (function calling)
- Streaming token generation

## Prerequisites

- CMake 3.20+
- C++20 compiler (GCC 10+, Clang 12+, MSVC 2019+)
- libnats-dev (optional)
- CUDA Toolkit 11.7+ (for GPU support)

## Building

### CPU Only (Development)

```bash
# Using build script
chmod +x build.sh
./build.sh

# Or manual build
mkdir build && cd build
cmake .. -DSOVALUNE_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### With CUDA

```bash
# Using build script
./build.sh --cuda

# Or manual build
mkdir build-cuda && cd build-cuda
cmake .. -DSOVALUNE_USE_CUDA=ON -DSOVALUNE_CPU_ONLY=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Debug Build

```bash
./build.sh --debug
```

## Running

### Basic Usage

```bash
export SOVALUNE_MODEL_PATH="/path/to/model.gguf"
export SOVALUNE_RUNTIME_MODE="cpu"
./build/sovalune-model-runtime
```

### With CUDA

```bash
export SOVALUNE_MODEL_PATH="/path/to/model.gguf"
export SOVALUNE_RUNTIME_MODE="cuda"
./build-cuda/sovalune-model-runtime
```

### Command Line Options

```
Usage: sovalune-model-runtime [OPTIONS]

Options:
  -h, --help              Show help message
  -v, --version           Show version
  -m, --model PATH        Path to GGUF model file
  -c, --context N         Context size (default: 4096)
  -t, --threads N         CPU threads (default: 4)
  --gpu-layers N          GPU layers (default: 99)
  --gpu-device N          GPU device ID (default: auto)
  --port N                REST API port (default: 8080)
  --no-nats               Disable NATS
  --verbose               Verbose output
```

## REST API

The runtime provides an OpenAI-compatible REST API.

### Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions

### Example Request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sovalune-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SOVALUNE_MODEL_PATH` | | Path to GGUF model file |
| `SOVALUNE_RUNTIME_MODE` | `cpu` | `cpu` or `cuda` |
| `SOVALUNE_NATS_URL` | `nats://localhost:4222` | NATS connection URL |
| `SOVALUNE_REST_PORT` | `8080` | REST API port |

## Architecture

```
src/
├── engine/               # Inference engine
│   ├── inference_engine  # Main engine wrapper
│   ├── llama_backend     # llama.cpp integration
│   ├── engine_config     # Configuration
│   └── gpu_manager       # CUDA device management
├── server/               # REST API
│   └── rest_api_server   # OpenAI-compatible API
├── tool_calling/         # Tool call parsing
├── nats_bridge/          # NATS messaging
├── tokenizer/            # Token counting
└── main.cpp              # Entry point
```

## Performance

### GPU Acceleration

For best performance with CUDA:
- Use models with GPU quantization (e.g., Q4_K_M, Q5_K_M)
- Set appropriate GPU layers (`--gpu-layers`)
- Enable flash attention for supported models

### CPU Optimization

- Use multiple threads (`--threads N`)
- Enable AVX2/AVX-512 if available
- Use quantized models for faster inference

## License

MIT
