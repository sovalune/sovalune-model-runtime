#include "llama_backend.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <cstring>
#include <chrono>

// llama.cpp headers
#ifdef __cplusplus
extern "C" {
#endif
#include "llama.h"
#ifdef __cplusplus
}
#endif

namespace sovalune {

LlamaBackend::LlamaBackend(const EngineConfig& config) : config_(config) {
    start_time_ = std::chrono::steady_clock::now();
    seed_ = config_.seed;
}

LlamaBackend::~LlamaBackend() {
    unload();
}

bool LlamaBackend::load_model(const std::string& model_path) {
    if (model_path.empty()) {
        std::cerr << "[LlamaBackend] Empty model path" << std::endl;
        return false;
    }

    std::cout << "[LlamaBackend] Loading model: " << model_path << std::endl;

    // Initialize llama backend
    llama_backend_init();

    // Model parameters
    auto model_params = llama_model_default_params();
    model_params.n_ctx = config_.n_ctx;
    model_params.n_batch = config_.n_batch;
    model_params.use_mmap = config_.use_mmap;
    model_params.use_mlock = config_.use_mlock;

    // GPU settings
    if (config_.mode == "cuda") {
        model_params.n_gpu_layers = config_.n_gpu_layers;
        std::cout << "[LlamaBackend] GPU layers: " << model_params.n_gpu_layers << std::endl;
    } else {
        model_params.n_gpu_layers = 0;
    }

    // Load model
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        std::cerr << "[LlamaBackend] Failed to load model from: " << model_path << std::endl;
        return false;
    }

    std::cout << "[LlamaBackend] Model loaded successfully" << std::endl;
    std::cout << "[LlamaBackend] Model name: " << model_name() << std::endl;
    std::cout << "[LlamaBackend] Vocab size: " << vocab_size() << std::endl;

    // Create context
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config_.n_ctx;
    ctx_params.n_batch = config_.n_batch;
    ctx_params.n_threads = config_.n_threads;
    ctx_params.n_threads_batch = config_.n_threads;

    if (config_.mode == "cuda") {
        ctx_params.flash_attn = config_.flash_attention;
    }

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        std::cerr << "[LlamaBackend] Failed to create context" << std::endl;
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    model_loaded_ = true;
    std::cout << "[LlamaBackend] Context created: " << n_ctx() << " tokens" << std::endl;
    std::cout << "[LlamaBackend] Threads: " << config_.n_threads << std::endl;

    if (seed_ >= 0) {
        llama_set_rng_seed(ctx_, seed_);
        std::cout << "[LlamaBackend] RNG seed: " << seed_ << std::endl;
    }

    return true;
}

void LlamaBackend::unload() {
    if (!model_loaded_) return;

    std::cout << "[LlamaBackend] Unloading model..." << std::endl;

    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }

    model_loaded_ = false;
    std::cout << "[LlamaBackend] Model unloaded" << std::endl;
}

std::vector<int32_t> LlamaBackend::tokenize(const std::string& text, bool add_bos) {
    if (!model_ || !model_loaded_) return {};

    // Estimate token count
    int n_tokens = llama_tokenize(model_, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_bos, true);

    if (n_tokens < 0) {
        std::cerr << "[LlamaBackend] Tokenization failed" << std::endl;
        return {};
    }

    std::vector<int32_t> tokens(n_tokens);
    llama_tokenize(model_, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), n_tokens, add_bos, true);

    return tokens;
}

std::string LlamaBackend::detokenize(const std::vector<int32_t>& tokens) {
    if (!model_ || tokens.empty()) return {};

    std::string result;
    result.reserve(tokens.size() * 2);

    for (int32_t token : tokens) {
        char buf[256];
        int n = llama_token_to_piece(model_, token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            result.append(buf, n);
        }
    }

    return result;
}

int32_t LlamaBackend::tokenize_single(const std::string& text) {
    auto tokens = tokenize(text, false);
    return tokens.empty() ? -1 : tokens[0];
}

std::string LlamaBackend::detokenize_single(int32_t token_id) {
    return detokenize({token_id});
}

bool LlamaBackend::is_eos(int32_t token_id) const {
    if (!model_) return false;
    return llama_token_is_eog(model_, token_id);
}

bool LlamaBackend::is_bos(int32_t token_id) const {
    if (!model_) return false;
    return llama_token_is_bos(model_, token_id);
}

int32_t LlamaBackend::generate_token(
    const std::vector<int32_t>& prompt_tokens,
    int32_t max_tokens
) {
    if (!ctx_ || prompt_tokens.empty()) return -1;

    // Create batch for prompt
    auto batch = llama_batch_init(static_cast<int32_t>(prompt_tokens.size()), 0, 1);

    // Add prompt tokens to batch
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        llama_batch_add(&batch, prompt_tokens[i], static_cast<int64_t>(i), {0}, false);
    }

    // Set logits for last token
    batch.logits[batch.n_tokens - 1] = true;

    // Decode
    if (llama_decode(ctx_, batch) != 0) {
        std::cerr << "[LlamaBackend] Failed to decode prompt" << std::endl;
        llama_batch_free(batch);
        return -1;
    }

    // Get logits for last token
    float* logits = llama_get_logits_ith(ctx_, batch.n_tokens - 1);
    int32_t n_vocab = llama_n_vocab(model_);

    // Sample token
    int32_t token = sample_token(
        logits, n_vocab,
        config_.temperature,
        config_.top_p,
        config_.top_k
    );

    llama_batch_free(batch);
    return token;
}

std::vector<int32_t> LlamaBackend::generate(
    const std::vector<int32_t>& prompt_tokens,
    int max_tokens,
    float temperature,
    float top_p,
    int top_k,
    std::function<void(int32_t)> on_token
) {
    if (!ctx_ || prompt_tokens.empty()) return {};

    std::vector<int32_t> generated;
    generated.reserve(max_tokens);

    // Reset state for new generation
    llama_kv_cache_clear(ctx_);

    // Process prompt tokens (prefill)
    int batch_size = config_.n_batch;
    for (size_t i = 0; i < prompt_tokens.size(); i += batch_size) {
        size_t end = std::min(i + static_cast<size_t>(batch_size), prompt_tokens.size());
        int len = static_cast<int>(end - i);

        auto batch = llama_batch_init(len, 0, 1);

        for (size_t j = 0; j < static_cast<size_t>(len); ++j) {
            llama_batch_add(&batch, prompt_tokens[i + j], static_cast<int64_t>(i + j), {0}, false);
        }

        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx_, batch) != 0) {
            std::cerr << "[LlamaBackend] Failed to decode prompt batch" << std::endl;
            llama_batch_free(batch);
            return generated;
        }

        llama_batch_free(batch);
    }

    // Get logits for first token after prompt
    float* logits = llama_get_logits_ith(ctx_, -1);
    int32_t n_vocab = llama_n_vocab(model_);

    // Generate tokens
    int32_t cur_token = sample_token(logits, temperature, top_p, top_k);

    for (int i = 0; i < max_tokens; ++i) {
        if (is_eos(cur_token)) break;

        generated.push_back(cur_token);

        if (on_token) {
            on_token(cur_token);
        }

        // Prepare next batch
        auto batch = llama_batch_init(1, 0, 1);
        llama_batch_add(&batch, cur_token, static_cast<int64_t>(prompt_tokens.size() + i), {0}, false);
        batch.logits[0] = true;

        if (llama_decode(ctx_, batch) != 0) {
            std::cerr << "[LlamaBackend] Failed to decode token " << i << std::endl;
            llama_batch_free(batch);
            break;
        }

        logits = llama_get_logits_ith(ctx_, 0);
        cur_token = sample_token(logits, temperature, top_p, top_k);

        llama_batch_free(batch);
    }

    return generated;
}

void LlamaBackend::reset() {
    if (ctx_) {
        llama_kv_cache_clear(ctx_);
    }
}

std::string LlamaBackend::model_name() const {
    if (!model_) return "unknown";
    return llama_model_name(model_);
}

int32_t LlamaBackend::vocab_size() const {
    if (!model_) return 0;
    return llama_n_vocab(model_);
}

int32_t LlamaBackend::n_ctx() const {
    if (!ctx_) return 0;
    return llama_n_ctx(ctx_);
}

int32_t LlamaBackend::n_embd() const {
    if (!model_) return 0;
    return llama_n_embd(model_);
}

int32_t LlamaBackend::n_layer() const {
    if (!model_) return 0;
    return llama_n_layer(model_);
}

int32_t LlamaBackend::n_head() const {
    if (!model_) return 0;
    return llama_n_head(model_);
}

std::string LlamaBackend::device_name() const {
    if (!model_) return "cpu";
#ifdef GGML_USE_CUBLAS
    if (config_.mode == "cuda") {
        return "cuda";
    }
#endif
    return "cpu";
}

int64_t LlamaBackend::device_memory() const {
    // TODO: Implement when llama.cpp exposes memory info
    return 0;
}

bool LlamaBackend::is_cuda() const {
#ifdef GGML_USE_CUBLAS
    return config_.mode == "cuda";
#else
    return false;
#endif
}

int64_t LlamaBackend::prompt_tokens_processed() const {
    return prompt_tokens_processed_.load();
}

int64_t LlamaBackend::tokens_generated() const {
    return tokens_generated_.load();
}

double LlamaBackend::avg_tokens_per_second() const {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (elapsed <= 0) return 0.0;
    return static_cast<double>(tokens_generated_.load()) / elapsed;
}

void LlamaBackend::set_seed(int seed) {
    seed_ = seed;
    if (ctx_) {
        llama_set_rng_seed(ctx_, seed);
    }
}

int32_t LlamaBackend::sample_token(
    float* logits,
    float temperature,
    float top_p,
    int top_k
) {
    int32_t n_vocab = llama_n_vocab(model_);

    if (temperature <= 0.0f || temperature < 1e-6f) {
        return sample_greedy(logits);
    }

    // Copy logits for modification
    std::vector<float> logits_copy(logits, logits + n_vocab);

    // Apply temperature
    apply_temperature(logits_copy.data(), n_vocab, temperature);

    // Apply top-k
    if (top_k > 0 && top_k < n_vocab) {
        apply_top_k(logits_copy.data(), n_vocab, top_k);
    }

    // Apply top-p
    return sample_top_p(logits_copy.data(), n_vocab, top_p);
}

int32_t LlamaBackend::sample_greedy(float* logits) {
    int32_t n_vocab = llama_n_vocab(model_);
    return static_cast<int32_t>(std::distance(logits, std::max_element(logits, logits + n_vocab)));
}

void LlamaBackend::apply_temperature(float* logits, int n_vocab, float temperature) {
    for (int i = 0; i < n_vocab; ++i) {
        logits[i] /= temperature;
    }
}

void LlamaBackend::apply_top_k(float* logits, int n_vocab, int k) {
    if (k >= n_vocab) return;

    // Create index array
    std::vector<int32_t> indices(n_vocab);
    std::iota(indices.begin(), indices.end(), 0);

    // Partial sort by logit value
    std::partial_sort(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [&logits](int32_t a, int32_t b) { return logits[a] > logits[b]; }
    );

    // Set logits outside top-k to -infinity
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<bool> in_top_k(n_vocab, false);
    for (int i = 0; i < k; ++i) {
        in_top_k[indices[i]] = true;
    }

    for (int i = 0; i < n_vocab; ++i) {
        if (!in_top_k[i]) {
            logits[i] = neg_inf;
        }
    }
}

int32_t LlamaBackend::sample_top_p(float* logits, int n_vocab, float p) {
    // Sort by logit value (descending)
    std::vector<int32_t> indices(n_vocab);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(
        indices.begin(),
        indices.end(),
        [&logits](int32_t a, int32_t b) { return logits[a] > logits[b]; }
    );

    // Convert to probabilities
    float max_logit = logits[indices[0]];
    std::vector<float> probs(n_vocab);
    float sum = 0.0f;

    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = std::exp(logits[indices[i]] - max_logit);
        sum += probs[i];
    }

    // Apply top-p filtering
    float cumsum = 0.0f;
    int32_t last_valid = indices[0];

    for (int i = 0; i < n_vocab; ++i) {
        cumsum += probs[i] / sum;
        if (cumsum >= p) {
            // Set all remaining to -inf
            for (int j = i + 1; j < n_vocab; ++j) {
                probs[indices[j]] = 0.0f;
            }
            break;
        }
        last_valid = indices[i];
    }

    // Sample from remaining distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());

    return indices[dist(gen)];
}

}  // namespace sovalune
