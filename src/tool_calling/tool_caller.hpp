#pragma once

#include <string>
#include <vector>
#include <functional>

namespace sovalune {

struct ToolCall {
    std::string tool_name;
    std::string arguments;
    std::string request_id;
};

using ToolCallHandler = std::function<std::string(const ToolCall&)>;

class ToolCaller {
public:
    ToolCaller();
    ~ToolCaller();
    
    // Register tool handler
    void register_tool(const std::string& tool_name, ToolCallHandler handler);
    
    // Parse and execute tool call from model output
    bool try_parse_tool_call(const std::string& output, ToolCall& result);
    
    // Execute tool call
    std::string execute_tool(const ToolCall& call);
    
private:
    std::unordered_map<std::string, ToolCallHandler> handlers_;
};

}  // namespace sovalune
