#include "inference_engine.hpp"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <thread>

namespace sovalune {

InferenceEngine::InferenceEngine(const EngineConfig& config)
    : config_(config),
      nats_(config.nats_url.empty() ? "nats://localhost:4222" : config.nats_url) {
}

InferenceEngine::~InferenceEngine() {
    stop();
}

bool InferenceEngine::start() {
    std::cout << "[ModelRuntime] Starting engine..." << std::endl;
    std::cout << "[ModelRuntime] Mode: " << config_.mode << std::endl;
    std::cout << "[ModelRuntime] Model: " << config_.model_path << std::endl;
    std::cout << "[ModelRuntime] Context: " << config_.n_ctx << std::endl;
    std::cout << "[ModelRuntime] Threads: " << config_.n_threads << std::endl;

    // Load tokenizer if available
    if (!config_.model_path.empty()) {
        std::string vocab_path = config_.model_path + ".vocab";
        if (tokenizer_.load(vocab_path)) {
            std::cout << "[ModelRuntime] Tokenizer loaded" << std::endl;
        } else {
            std::cout << "[ModelRuntime] No tokenizer found, using default" << std::endl;
        }
    }

    // Register tool handlers
    tool_caller_.register_tool("memory_search", [](const ToolCall& call) -> std::string {
        return R"({"status": "ok", "results": []})";
    });
    tool_caller_.register_tool("memory_write", [](const ToolCall& call) -> std::string {
        return R"({"status": "ok", "id": "memory_001"})";
    });
    tool_caller_.register_tool("code_execute", [](const ToolCall& call) -> std::string {
        return R"({"status": "ok", "output": "executed"})";
    });

    ready_ = true;
    running_ = true;

    // Start NATS subscription worker
    if (nats_.is_connected()) {
        nats_.subscribe_inference([this](const InferenceResponse& response) {
            // Handle inference requests from NATS
        });
        std::cout << "[ModelRuntime] Connected to NATS, listening for requests" << std::endl;
    } else {
        std::cout << "[ModelRuntime] Running without NATS (standalone mode)" << std::endl;
    }

    std::cout << "[ModelRuntime] Engine ready" << std::endl;
    return true;
}

void InferenceEngine::stop() {
    if (!running_) return;

    running_ = false;
    ready_ = false;

    if (worker_.joinable()) {
        worker_.join();
    }

    std::cout << "[ModelRuntime] Engine stopped" << std::endl;
}

std::string InferenceEngine::generate(
    const std::string& prompt,
    const GenerationConfig& gen_config
) {
    if (!ready_) {
        throw std::runtime_error("Engine not ready");
    }

    // Tokenize prompt
    auto prompt_tokens = tokenizer_.encode(prompt);

    // Generate tokens
    auto response_tokens = sample_tokens(
        prompt_tokens,
        gen_config.max_tokens,
        gen_config.temperature,
        gen_config.top_p
    );

    // Decode response
    return tokenizer_.decode(response_tokens);
}

void InferenceEngine::generate_stream(
    const std::string& prompt,
    TokenCallback on_token,
    const GenerationConfig& gen_config
) {
    if (!ready_) {
        throw std::runtime_error("Engine not ready");
    }

    // Tokenize prompt
    auto prompt_tokens = tokenizer_.encode(prompt);

    // Generate tokens one by one
    auto response_tokens = sample_tokens(
        prompt_tokens,
        gen_config.max_tokens,
        gen_config.temperature,
        gen_config.top_p
    );

    // Stream tokens via callback
    for (int token_id : response_tokens) {
        Token token;
        token.id = token_id;
        token.text = tokenizer_.decode({token_id});
        token.probability = 1.0f;
        on_token(token);

        // Small delay to simulate streaming
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

std::string InferenceEngine::generate_with_tools(
    const std::string& prompt,
    ToolCallCallback on_tool_call,
    const GenerationConfig& gen_config
) {
    if (!ready_) {
        throw std::runtime_error("Engine not ready");
    }

    std::string full_response;
    std::string current_prompt = prompt;
    int iterations = 0;

    while (iterations < config_.max_tool_calls) {
        // Generate response
        auto response = generate(current_prompt, gen_config);
        full_response += response;

        // Check for tool calls in response
        ToolCall tool_call;
        if (tool_caller_.try_parse_tool_call(response, tool_call)) {
            on_tool_call(tool_call.tool_name, tool_call.arguments);

            // Execute tool
            auto result = tool_caller_.execute_tool(tool_call);

            // Add tool result to context for next iteration
            current_prompt += "\n\nTool result: " + result;
            iterations++;
        } else {
            // No tool call - we're done
            break;
        }
    }

    return full_response;
}

bool InferenceEngine::is_ready() const {
    return ready_;
}

std::string InferenceEngine::get_model_name() const {
    return config_.model_path.empty() ? "stub" : config_.model_path;
}

int InferenceEngine::get_context_size() const {
    return config_.n_ctx;
}

std::vector<int> InferenceEngine::sample_tokens(
    const std::vector<int>& prompt_tokens,
    int max_tokens,
    float temperature,
    float top_p
) {
    // In production, this would use the actual model for inference.
    // For now, generate a placeholder response.
    std::vector<int> tokens;

    // Simple response generation (stub)
    std::string response = "I am Sovalune AI, ready to help with your software engineering tasks.";
    auto response_tokens = tokenizer_.encode(response);

    // Limit to max_tokens
    int count = 0;
    for (int t : response_tokens) {
        if (count >= max_tokens) break;
        tokens.push_back(t);
        count++;
    }

    return tokens;
}

}  // namespace sovalune
