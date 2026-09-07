cmake_minimum_required(VERSION 3.20)
project(sovalune-model-runtime VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(SOVALUNE_USE_CUDA "Enable CUDA support" OFF)
option(SOVALUNE_CPU_ONLY "CPU only mode (dev)" ON)
option(SOVALUNE_USE_LLAMA_CPP "Use llama.cpp backend" ON)
option(SOVALUNE_BUILD_TESTS "Build tests" OFF)
option(SOVALUNE_BUILD_SHARED "Build shared library" OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# llama.cpp integration
if(SOVALUNE_USE_LLAMA_CPP)
    # Fetch llama.cpp if not found
    include(FetchContent)
    
    # Options for llama.cpp build
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    
    if(SOVALUNE_USE_CUDA)
        set(LLAMA_CUBLAS ON CACHE BOOL "" FORCE)
        set(GGML_CUDA ON CACHE BOOL "" FORCE)
        message(STATUS "Building llama.cpp with CUDA support")
    endif()
    
    if(SOVALUNE_CPU_ONLY)
        set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
        set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
    endif()
    
    FetchContent_Declare(
        llama
        GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    
    FetchContent_MakeAvailable(llama)
    
    set(LLAMA_INCLUDE_DIRS ${llama_SOURCE_DIR}/include ${llama_SOURCE_DIR})
    set(LLAMA_LIBRARIES llama ggml common)
    
    add_definitions(-DSOVALUNE_HAS_LLAMA_CPP=1)
    message(STATUS "llama.cpp integrated via FetchContent")
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/engine/inference_engine.cpp
    src/engine/engine_config.cpp
    src/engine/llama_backend.cpp
    src/engine/gpu_manager.cpp
    src/server/rest_api_server.cpp
    src/tool_calling/tool_caller.cpp
    src/tool_calling/json_parser.cpp
    src/nats_bridge/nats_client.cpp
    src/nats_bridge/message_handler.cpp
    src/tokenizer/tokenizer.cpp
)

# Headers
set(HEADERS
    src/engine/inference_engine.hpp
    src/engine/engine_config.hpp
    src/engine/llama_backend.hpp
    src/engine/gpu_manager.hpp
    src/server/rest_api_server.hpp
    src/tool_calling/tool_caller.hpp
    src/tool_calling/json_parser.hpp
    src/nats_bridge/nats_client.hpp
    src/nats_bridge/message_handler.hpp
    src/tokenizer/tokenizer.hpp
)

# Create executable
add_executable(sovalune-model-runtime ${SOURCES} ${HEADERS})

target_include_directories(sovalune-model-runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LLAMA_INCLUDE_DIRS}
)

# Compile definitions
target_compile_definitions(sovalune-model-runtime PRIVATE
    $<$<BOOL:${SOVALUNE_USE_LLAMA_CPP}>:SOVALUNE_HAS_LLAMA_CPP>
    $<$<BOOL:${SOVALUNE_USE_CUDA}>:SOVALUNE_CUDA_ENABLED>
    $<$<BOOL:${SOVALUNE_CPU_ONLY}>:SOVALUNE_CPU_ONLY>
)

# Link libraries
if(NOT WIN32)
    target_link_libraries(sovalune-model-runtime PRIVATE pthread)
else()
    target_link_libraries(sovalune-model-runtime PRIVATE ws2_32)
endif()

# NATS
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(NATS IMPORTED_TARGET libnats)
    if(NATS_FOUND)
        target_link_libraries(sovalune-model-runtime PRIVATE PkgConfig::NATS)
        target_compile_definitions(sovalune-model-runtime PRIVATE SOVALUNE_HAS_NATS=1)
    else()
        message(Warning "libnats not found, building without NATS support")
    endif()
endif()

# llama.cpp
if(SOVALUNE_USE_LLAMA_CPP)
    target_link_libraries(sovalune-model-runtime PRIVATE ${LLAMA_LIBRARIES})
endif()

# CUDA
if(SOVALUNE_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(sovalune-model-runtime PRIVATE CUDA::cudart CUDA::cublas)
    
    # Additional CUDA libraries for llama.cpp
    if(GGML_CUDA)
        target_link_libraries(sovalune-model-runtime PRIVATE ggml-cuda)
    endif()
endif()

# Platform-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(NOT SOVALUNE_CPU_ONLY)
        target_compile_options(sovalune-model-runtime PRIVATE -mavx2 -mfma -mf16c)
    endif()
endif()

# Install
install(TARGETS sovalune-model-runtime DESTINATION bin)

# Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
