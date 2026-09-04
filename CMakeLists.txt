cmake_minimum_required(VERSION 3.20)
project(sovalune-model-runtime VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(SOVALUNE_USE_CUDA "Enable CUDA support" OFF)
option(SOVALUNE_CPU_ONLY "CPU only mode (dev)" ON)

# Dependencies - NATS is optional
find_package(PkgConfig)
pkg_check_modules(NATS IMPORTED_TARGET libnats)
if(NOT NATS_FOUND)
    message(Warning "libnats not found, building without NATS support")
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/engine/inference_engine.cpp
    src/engine/engine_config.cpp
    src/tool_calling/tool_caller.cpp
    src/tool_calling/json_parser.cpp
    src/nats_bridge/nats_client.cpp
    src/nats_bridge/message_handler.cpp
    src/tokenizer/tokenizer.cpp
)

# Create executable
add_executable(sovalune-model-runtime ${SOURCES})

target_include_directories(sovalune-model-runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(sovalune-model-runtime PRIVATE
    pthread
)
if(NATS_FOUND)
    target_link_libraries(sovalune-model-runtime PRIVATE PkgConfig::NATS)
endif()

# CUDA support
if(SOVALUNE_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(sovalune-model-runtime PRIVATE SOVALUNE_CUDA_ENABLED)
    target_link_libraries(sovalune-model-runtime PRIVATE CUDA::cudart CUDA::cublas)
endif()

# CPU only mode
if(SOVALUNE_CPU_ONLY)
    target_compile_definitions(sovalune-model-runtime PRIVATE SOVALUNE_CPU_ONLY)
endif()

# Install
install(TARGETS sovalune-model-runtime DESTINATION bin)
