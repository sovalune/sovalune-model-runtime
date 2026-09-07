#include "inference_engine.hpp"
#include "gpu_manager.hpp"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <future>

namespace sovalune {

InferenceEngine::InferenceEngine(const EngineConfig& config)
    : config_(config),
      backend_(config),
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

    // Initialize GPU if CUDA mode
    if (config_.mode == "cuda") {
        if (!GpuManager::initialize()) {
            std::cerr << "[ModelRuntime] Failed to initialize GPU" << std::endl;
            return false;
        }

        // Auto-select GPU device if not specified
        if (config_.gpu_device < 0) {
            config_.gpu_device = GpuManager::select_best_device();
            if (config_.gpu_device < 0) {
                std::cerr << "[ModelRuntime] No suitable GPU found" << std::endl;
                return false;
            }
        }

        GpuManager::set_device(config_.gpu_device);
        std::cout << "[ModelRuntime] Using GPU: " << GpuManager::device_name(config_.gpu_device) << std::endl;
        GpuManager::print_device_info(config_.gpu_device);
    }

    // Load model via LlamaBackend
    if (!config_.model_path.empty()) {
        if (!backend_.load_model(config_.model_path)) {
            std::cerr << "[ModelRuntime] Failed to load model" << std::endl;
            return false;
        }

        // Tokenizer is handled by llama.cpp internally, but we can load custom vocab
        std::string vocab_path = config_.model_path + ".vocab";
        if (tokenizer_.load(vocab_path)) {
            std::cout << "[ModelRuntime] Custom tokenizer loaded" << std::endl;
        }
    } else {
        std::cout << "[ModelRuntime] No model path specified, running in stub mode" << std::endl;
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

    // Start worker thread for async processing
    worker_ = std::thread(&InferenceEngine::worker_loop, this);

    // Start NATS subscription
    if (nats_.is_connected()) {
        nats_.subscribe_inference([this](const InferenceResponse& response) {
            // Handle inference requests from NATS
        });
        std::cout << "[ModelRuntime] Connected to NATS, listening for requests" << std::endl;
    } else {
        std::cout << "[ModelRuntime] Running without NATS (standalone mode)" << std::endl;
    }

    std::cout << "[ModelRuntime] Engine ready" << std::endl;
    if (!config_.model_path.empty()) {
        std::cout << "[ModelRuntime] Model: " << backend_.model_name() << std::endl;
        std::cout << "[ModelRuntime] Vocab: " << backend_.vocab_size() << std::endl;
        std::cout << "[ModelRuntime] Device: " << backend_.device_name() << std::endl;
    }
    return true;
}

void InferenceEngine::stop() {
    if (!running_) return;

    running_ = false;
    ready_ = false;

    // Notify worker to stop
    queue_cv_.notify_all();

    if (worker_.joinable()) {
        worker_.join();
    }

    backend_.unload();

    // Shutdown GPU manager
    if (config_.mode == "cuda") {
        GpuManager::shutdown();
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

    generating_ = true;

    // Tokenize prompt
    std::vector<int32_t> prompt_tokens;
    if (backend_.vocab_size() > 0) {
        prompt_tokens = backend_.tokenize(prompt, true);
    } else {
        auto int_tokens = tokenizer_.encode(prompt);
        prompt_tokens.assign(int_tokens.begin(), int_tokens.end());
    }

    // Generate tokens
    auto response_tokens = sample_tokens(
        prompt_tokens,
        gen_config.max_tokens,
        gen_config.temperature,
        gen_config.top_p,
        gen_config.top_k
    );

    generating_ = false;

    // Decode response
    if (backend_.vocab_size() > 0) {
        return backend_.detokenize(response_tokens);
    } else {
        std::vector<int> int_tokens(response_tokens.begin(), response_tokens.end());
        return tokenizer_.decode(int_tokens);
    }
}

void InferenceEngine::generate_stream(
    const std::string& prompt,
    TokenCallback on_token,
    const GenerationConfig& gen_config
) {
    if (!ready_) {
        throw std::runtime_error("Engine not ready");
    }

    generating_ = true;

    // Tokenize prompt
    std::vector<int32_t> prompt_tokens;
    if (backend_.vocab_size() > 0) {
        prompt_tokens = backend_.tokenize(prompt, true);
    } else {
        auto int_tokens = tokenizer_.encode(prompt);
        prompt_tokens.assign(int_tokens.begin(), int_tokens.end());
    }

    // Generate tokens one by one with streaming
    std::function<void(int32_t)> stream_callback = [&](int32_t token_id) {
        Token token;
        token.id = token_id;
        token.text = token_to_text(token_id);
        token.probability = 1.0f;
        token.is_eos = backend_.is_eos(token_id);
        on_token(token);
    };

    if (backend_.vocab_size() > 0) {
        backend_.generate(
            prompt_tokens,
            gen_config.max_tokens,
            gen_config.temperature,
            gen_config.top_p,
            gen_config.top_k,
            stream_callback
        );
    } else {
        // Stub mode
        auto response_tokens = sample_tokens(
            prompt_tokens,
            gen_config.max_tokens,
            gen_config.temperature,
            gen_config.top_p,
            gen_config.top_k
        );

        for (int32_t token_id : response_tokens) {
            stream_callback(token_id);
        }
    }

    generating_ = false;
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

std::future<std::string> InferenceEngine::generate_async(
    const std::string& prompt,
    const GenerationConfig& gen_config
) {
    return std::async(std::launch::async, [this, prompt, gen_config]() {
        return generate(prompt, gen_config);
    });
}

void InferenceEngine::cancel_generation() {
    generating_ = false;
    backend_.reset();
}

bool InferenceEngine::is_ready() const {
    return ready_;
}

std::string InferenceEngine::get_model_name() const {
    if (!config_.model_path.empty()) {
        return backend_.model_name();
    }
    return "stub";
}

int InferenceEngine::get_context_size() const {
    return config_.n_ctx;
}

std::string InferenceEngine::get_device_name() const {
    return backend_.device_name();
}

bool InferenceEngine::is_cuda() const {
    return backend_.is_cuda();
}

double InferenceEngine::get_tokens_per_second() const {
    return backend_.avg_tokens_per_second();
}

int64_t InferenceEngine::get_total_tokens_generated() const {
    return backend_.tokens_generated();
}

void InferenceEngine::process_request(const InferenceRequest& request) {
    std::cout << "[ModelRuntime] Processing request: " << request.request_id << std::endl;

    GenerationConfig gen_config;
    gen_config.max_tokens = request.max_tokens;
    gen_config.temperature = request.temperature;

    auto response = generate(request.prompt_context, gen_config);

    // Publish response via NATS
    InferenceResponse nats_response;
    nats_response.request_id = request.request_id;
    nats_response.delta = response;
    nats_response.done = true;

    nats_.publish_response(nats_response);
}

void InferenceEngine::worker_loop() {
    while (running_) {
        InferenceTask task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !task_queue_.empty() || !running_;
            });

            if (!running_ && task_queue_.empty()) break;
            if (task_queue_.empty()) continue;

            task = std::move(task_queue_.front());
            task_queue_.pop();
        }

        process_task(task);
    }
}

void InferenceEngine::process_task(InferenceTask& task) {
    std::cout << "[ModelRuntime] Processing async task: " << task.request_id << std::endl;

    if (task.is_streaming && task.on_token) {
        generate_stream(task.prompt, task.on_token, task.gen_config);
    } else {
        auto result = generate(task.prompt, task.gen_config);

        // Publish result via NATS
        InferenceResponse response;
        response.request_id = task.request_id;
        response.delta = result;
        response.done = true;
        nats_.publish_response(response);
    }
}

std::vector<int32_t> InferenceEngine::sample_tokens(
    const std::vector<int32_t>& prompt_tokens,
    int max_tokens,
    float temperature,
    float top_p,
    int top_k
) {
    if (backend_.vocab_size() > 0) {
        // Use real backend
        return backend_.generate(prompt_tokens, max_tokens, temperature, top_p, top_k);
    }

    // Stub mode - generate placeholder response
    std::vector<int32_t> tokens;

    std::string response = "I am Sovalune AI, ready to help with your software engineering tasks.";
    std::vector<int32_t> response_tokens;

    if (backend_.vocab_size() > 0) {
        response_tokens = backend_.tokenize(response, false);
    } else {
        auto int_tokens = tokenizer_.encode(response);
        response_tokens.assign(int_tokens.begin(), int_tokens.end());
    }

    // Limit to max_tokens
    int count = 0;
    for (int32_t t : response_tokens) {
        if (count >= max_tokens) break;
        tokens.push_back(t);
        count++;
    }

    return tokens;
}

std::string InferenceEngine::token_to_text(int32_t token_id) {
    if (backend_.vocab_size() > 0) {
        return backend_.detokenize_single(token_id);
    }
    return tokenizer_.decode({token_id});
}

}  // namespace sovalune
