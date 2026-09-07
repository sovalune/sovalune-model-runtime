#pragma once

#include "engine_config.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>

namespace sovalune {

/// Forward declarations - llama.cpp types handled via void* for ABI safety
struct llama_context;
struct llama_model;
struct llama_token_data_array;

/// llama.cpp backend wrapper for GGUF model inference.
///
/// Handles:
/// - Model loading with configurable parameters
/// - Context management and KV cache
/// - Token sampling (greedy, top-k, top-p, temperature)
/// - CUDA acceleration via cuBLAS
/// - Batch inference for throughput
class LlamaBackend {
public:
    explicit LlamaBackend(const EngineConfig& config);
    ~LlamaBackend();

    LlamaBackend(const LlamaBackend&) = delete;
    LlamaBackend& operator=(const LlamaBackend&) = delete;

    /// Load model from GGUF file
    bool load_model(const std::string& model_path);

    /// Free model and context
    void unload();

    /// Tokenize text using model's built-in tokenizer
    std::vector<int32_t> tokenize(const std::string& text, bool add_bos = true);

    /// Detokenize token IDs to text
    std::string detokenize(const std::vector<int32_t>& tokens);

    /// Tokenize a single piece of text
    int32_t tokenize_single(const std::string& text);

    /// Detokenize a single token
    std::string detokenize_single(int32_t token_id);

    /// Check if token is end of sequence
    bool is_eos(int32_t token_id) const;

    /// Check if token is beginning of sequence
    bool is_bos(int32_t token_id) const;

    /// Generate a single token given prompt tokens (prefill + decode)
    int32_t generate_token(const std::vector<int32_t>& prompt_tokens, int32_t max_tokens = 1);

    /// Generate tokens with streaming callback
    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt_tokens,
        int max_tokens,
        float temperature,
        float top_p,
        int top_k,
        std::function<void(int32_t)> on_token = nullptr
    );

    /// Reset KV cache for new sequence
    void reset();

    /// Get model info
    std::string model_name() const;
    int32_t vocab_size() const;
    int32_t n_ctx() const;
    int32_t n_embd() const;
    int32_t n_layer() const;
    int32_t n_head() const;

    /// Get device info
    std::string device_name() const;
    int64_t device_memory() const;
    bool is_cuda() const;

    /// Get performance stats
    int64_t prompt_tokens_processed() const;
    int64_t tokens_generated() const;
    double avg_tokens_per_second() const;

    /// Set RNG seed
    void set_seed(int seed);

    /// Get internal context pointer (for advanced usage)
    llama_context* raw_context() const { return ctx_; }

private:
    EngineConfig config_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;

    // State
    std::atomic<bool> model_loaded_{false};
    int32_t seed_ = -1;

    // Stats
    std::atomic<int64_t> prompt_tokens_processed_{0};
    std::atomic<int64_t> tokens_generated_{0};
    std::chrono::steady_clock::time_point start_time_;

    /// Sample next token given logits
    int32_t sample_token(
        float* logits,
        float temperature,
        float top_p,
        int top_k
    );

    /// Apply greedy sampling
    int32_t sample_greedy(float* logits);

    /// Apply temperature scaling
    void apply_temperature(float* logits, int n_vocab, float temperature);

    /// Apply top-k filtering
    void apply_top_k(float* logits, int n_vocab, int k);

    /// Apply top-p (nucleus) sampling
    int32_t sample_top_p(float* logits, int n_vocab, float p);
};

}  // namespace sovalune
