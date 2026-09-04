#pragma once

#include "engine_config.hpp"
#include "../nats_bridge/nats_client.hpp"
#include "../tokenizer/tokenizer.hpp"
#include "../tool_calling/tool_caller.hpp"
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <atomic>
#include <thread>

namespace sovalune {

struct Token {
    int id;
    std::string text;
    float probability;
};

using TokenCallback = std::function<void(const Token&)>;
using ToolCallCallback = std::function<void(const std::string& tool_name, const std::string& arguments)>;

/// Production inference engine that connects to NATS and processes requests.
///
/// Architecture:
/// 1. Connects to NATS server
/// 2. Subscribes to inference.requests
/// 3. For each request: tokenize prompt, run model inference, stream tokens
/// 4. Publishes response tokens back via NATS
class InferenceEngine {
public:
    explicit InferenceEngine(const EngineConfig& config);
    ~InferenceEngine();

    // Non-copyable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    // Start the engine (connects to NATS, loads model)
    bool start();

    // Stop the engine
    void stop();

    // Generate response (blocking)
    std::string generate(
        const std::string& prompt,
        const GenerationConfig& gen_config = GenerationConfig{}
    );

    // Stream response (callback-based)
    void generate_stream(
        const std::string& prompt,
        TokenCallback on_token,
        const GenerationConfig& gen_config = GenerationConfig{}
    );

    // Generate with tool calling
    std::string generate_with_tools(
        const std::string& prompt,
        ToolCallCallback on_tool_call,
        const GenerationConfig& gen_config = GenerationConfig{}
    );

    // Health check
    bool is_ready() const;

    // Get info
    std::string get_model_name() const;
    int get_context_size() const;

    // Access components
    NatsClient& nats() { return nats_; }
    Tokenizer& tokenizer() { return tokenizer_; }
    ToolCaller& tool_caller() { return tool_caller_; }

private:
    EngineConfig config_;
    NatsClient nats_;
    Tokenizer tokenizer_;
    ToolCaller tool_caller_;
    std::atomic<bool> ready_{false};
    std::atomic<bool> running_{false};
    std::thread worker_;

    // Process a single inference request
    void process_request(const InferenceRequest& request);

    // Token generation (simplified - in production uses actual model)
    std::vector<int> sample_tokens(
        const std::vector<int>& prompt_tokens,
        int max_tokens,
        float temperature,
        float top_p
    );
};

}  // namespace sovalune
