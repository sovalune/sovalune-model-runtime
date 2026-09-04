#pragma once

#include "engine_config.hpp"
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace sovalune {

struct Token {
    int id;
    std::string text;
    float probability;
};

using TokenCallback = std::function<void(const Token&)>;
using ToolCallCallback = std::function<void(const std::string& tool_name, const std::string& arguments)>;

class InferenceEngine {
public:
    explicit InferenceEngine(const EngineConfig& config);
    ~InferenceEngine();
    
    // Non-copyable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    
    // Generate response
    std::string generate(
        const std::string& prompt,
        const GenerationConfig& gen_config = GenerationConfig{}
    );
    
    // Stream response
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
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sovalune
