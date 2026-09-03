# Build
mkdir -p build
cd build
cmake .. -DSOVALUNE_CPU_ONLY=ON
make -j$(nproc)

# Run
export SOVALUNE_MODEL_PATH="/path/to/model.gguf"
export SOVALUNE_RUNTIME_MODE="cpu"
export SOVALUNE_NATS_URL="nats://localhost:4222"
./sovalune-model-runtime
