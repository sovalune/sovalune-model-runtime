#pragma once

#include "engine_config.hpp"
#include "llama_backend.hpp"
#include "../nats_bridge/nats_client.hpp"
#include "../tokenizer/tokenizer.hpp"
#include "../tool_calling/tool_caller.hpp"
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace sovalune {

struct Token {
    int32_t id;
    std::string text;
    float probability;
    bool is_eos;
};

using TokenCallback = std::function<void(const Token&)>;
using ToolCallCallback = std::function<void(const std::string& tool_name, const std::string& arguments)>;

/// Inference request for async processing
struct InferenceTask {
    std::string request_id;
    std::string prompt;
    GenerationConfig gen_config;
    TokenCallback on_token;
    bool is_streaming;
};

/// Production inference engine with llama.cpp backend.
///
/// Architecture:
/// 1. Loads GGUF model via LlamaBackend
/// 2. Connects to NATS for distributed inference
/// 3. Processes requests with streaming support
/// 4. Handles tool calling loops
class InferenceEngine {
public:
    explicit InferenceEngine(const EngineConfig& config);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /// Start the engine (loads model, connects to NATS)
    bool start();

    /// Stop the engine
    void stop();

    /// Generate response (blocking)
    std::string generate(
        const std::string& prompt,
        const GenerationConfig& gen_config = GenerationConfig{}
    );

    /// Stream response (callback-based)
    void generate_stream(
        const std::string& prompt,
        TokenCallback on_token,
        const GenerationConfig& gen_config = GenerationConfig{}
    );

    /// Generate with tool calling
    std::string generate_with_tools(
        const std::string& prompt,
        ToolCallCallback on_tool_call,
        const GenerationConfig& gen_config = GenerationConfig{}
    );

    /// Async generation (returns future)
    std::future<std::string> generate_async(
        const std::string& prompt,
        const GenerationConfig& gen_config = GenerationConfig{}
    );

    /// Cancel current generation
    void cancel_generation();

    /// Health check
    bool is_ready() const;

    /// Get info
    std::string get_model_name() const;
    int get_context_size() const;
    std::string get_device_name() const;
    bool is_cuda() const;

    /// Get performance stats
    double get_tokens_per_second() const;
    int64_t get_total_tokens_generated() const;

    /// Access components
    NatsClient& nats() { return nats_; }
    Tokenizer& tokenizer() { return tokenizer_; }
    ToolCaller& tool_caller() { return tool_caller_; }
    LlamaBackend& backend() { return backend_; }

private:
    EngineConfig config_;
    LlamaBackend backend_;
    NatsClient nats_;
    Tokenizer tokenizer_;
    ToolCaller tool_caller_;

    // State
    std::atomic<bool> ready_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> generating_{false};
    std::thread worker_;

    // Task queue
    std::queue<InferenceTask> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Process a single inference request
    void process_request(const InferenceRequest& request);

    // Worker thread for async processing
    void worker_loop();

    // Process a task from queue
    void process_task(InferenceTask& task);

    // Token generation using backend
    std::vector<int32_t> sample_tokens(
        const std::vector<int32_t>& prompt_tokens,
        int max_tokens,
        float temperature,
        float top_p,
        int top_k
    );

    // Convert token to text with special handling
    std::string token_to_text(int32_t token_id);
};

}  // namespace sovalune
