#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace sovalune {

/// Generation configuration for inference
struct GenerationConfig {
    int max_tokens = 1024;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 50;
    bool stream = true;
    int seed = -1;              // -1 for random
    bool stop_on_eos = true;
    std::vector<std::string> stop_sequences;
};

/// Engine configuration
struct EngineConfig {
    // Model settings
    std::string model_path;
    std::string mode = "cpu";           // "cpu" or "cuda"
    int n_ctx = 4096;                   // Context window size
    int n_batch = 512;                  // Batch size for prompt processing
    int n_threads = 4;                  // CPU threads
    bool use_mmap = true;              // Memory-mapped I/O
    bool use_mlock = false;            // Lock model in RAM

    // GPU settings
    int n_gpu_layers = 99;             // Layers to offload to GPU (99 = all)
    bool flash_attention = true;       // Use flash attention if available
    int gpu_device = -1;               // GPU device ID (-1 = auto)

    // Sampling
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 50;
    int seed = -1;                     // -1 for random

    // NATS connection
    std::string nats_url = "nats://localhost:4222";

    // REST API
    int rest_port = 8080;

    // Tool calling
    bool enable_tool_calling = true;
    int max_tool_calls = 5;
    int tool_call_timeout_ms = 30000;

    // Performance
    int max_concurrent_requests = 4;
    bool enable_kv_cache_reuse = true;
    int kv_cache_size = 0;             // 0 = auto

    // Logging
    bool verbose = false;
    bool log_tokens = false;
};

}  // namespace sovalune
