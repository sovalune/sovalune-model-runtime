#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <regex>
#include <iostream>

namespace sovalune {

struct ToolCall {
    std::string tool_name;
    std::string arguments;
    std::string request_id;
};

using ToolCallHandler = std::function<std::string(const ToolCall&)>;

class ToolCaller {
public:
    ToolCaller() = default;
    ~ToolCaller() = default;

    void register_tool(const std::string& tool_name, ToolCallHandler handler) {
        handlers_[tool_name] = handler;
    }

    bool try_parse_tool_call(const std::string& output, ToolCall& result) {
        std::regex tool_pattern(R"delimiter(\{"tool"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})\})delimiter");
        std::smatch matches;
        if (std::regex_search(output, matches, tool_pattern)) {
            result.tool_name = matches[1].str();
            result.arguments = matches[2].str();
            return true;
        }
        return false;
    }

    std::string execute_tool(const ToolCall& call) {
        auto it = handlers_.find(call.tool_name);
        if (it == handlers_.end()) {
            return "{\"error\": \"Unknown tool: " + call.tool_name + "\"}";
        }
        return it->second(call);
    }

private:
    std::unordered_map<std::string, ToolCallHandler> handlers_;
};

}  // namespace sovalune
