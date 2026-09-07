#!/bin/bash

# Sovalune Model Runtime build script

set -e

# Parse arguments
BUILD_TYPE="Release"
CUDA_SUPPORT="OFF"
CPU_ONLY="ON"
BUILD_DIR="build"

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --cuda)
            CUDA_SUPPORT="ON"
            CPU_ONLY="OFF"
            shift
            ;;
        --clean)
            rm -rf build build-cuda
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build directory
if [ "$CUDA_SUPPORT" = "ON" ]; then
    BUILD_DIR="build-cuda"
fi

echo "=== Building Sovalune Model Runtime ==="
echo "Build type: $BUILD_TYPE"
echo "CUDA: $CUDA_SUPPORT"
echo "CPU only: $CPU_ONLY"
echo "Build dir: $BUILD_DIR"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DSOVALUNE_USE_CUDA=$CUDA_SUPPORT \
    -DSOVALUNE_CPU_ONLY=$CPU_ONLY

# Build
cmake --build . -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "=== Build complete ==="
echo "Binary: $BUILD_DIR/sovalune-model-runtime"
echo ""

# Print usage
echo "Usage:"
echo "  export SOVALUNE_MODEL_PATH=\"/path/to/model.gguf\""
if [ "$CUDA_SUPPORT" = "ON" ]; then
    echo "  export SOVALUNE_RUNTIME_MODE=\"cuda\""
else
    echo "  export SOVALUNE_RUNTIME_MODE=\"cpu\""
fi
echo "  ./$BUILD_DIR/sovalune-model-runtime"
