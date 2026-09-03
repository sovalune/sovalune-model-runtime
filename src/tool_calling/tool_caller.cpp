#include "tool_caller.hpp"
#include <regex>
#include <iostream>

namespace sovalune {

ToolCaller::ToolCaller() = default;
ToolCaller::~ToolCaller() = default;

void ToolCaller::register_tool(const std::string& tool_name, ToolCallHandler handler) {
    handlers_[tool_name] = handler;
}

bool ToolCaller::try_parse_tool_call(const std::string& output, ToolCall& result) {
    // Look for tool call pattern: {"tool": "...", "arguments": {...}}
    std::regex tool_pattern(R"(\{"tool"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})\})");
    std::smatch matches;
    
    if (std::regex_search(output, matches, tool_pattern)) {
        result.tool_name = matches[1].str();
        result.arguments = matches[2].str();
        return true;
    }
    
    return false;
}

std::string ToolCaller::execute_tool(const ToolCall& call) {
    auto it = handlers_.find(call.tool_name);
    if (it == handlers_.end()) {
        return "{\"error\": \"Unknown tool: " + call.tool_name + "\"}";
    }
    
    return it->second(call);
}

}  // namespace sovalune
