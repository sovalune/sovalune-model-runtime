#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace sovalune {

struct GenerationConfig {
    int max_tokens = 1024;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 50;
    bool stream = true;
};

struct EngineConfig {
    std::string model_path;
    std::string mode = "cpu";  // "cpu" or "cuda"
    int n_ctx = 4096;
    int n_threads = 4;
    bool use_mmap = true;
    bool use_mlock = false;

    // NATS connection
    std::string nats_url = "nats://localhost:4222";

    // Tool calling
    bool enable_tool_calling = true;
    int max_tool_calls = 5;
    int tool_call_timeout_ms = 30000;
};

}  // namespace sovalune
