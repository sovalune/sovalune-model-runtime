#pragma once

#include <string>
#include <optional>

namespace sovalune {

struct ParsedToolCall {
    std::string name;
    std::string arguments;
};

class JsonParser {
public:
    static std::optional<ParsedToolCall> parse_tool_call(const std::string& json_str);
    static std::string serialize_response(const std::string& request_id, const std::string& content, bool done);
    static std::string serialize_tool_result(const std::string& request_id, const std::string& tool_name, const std::string& result);
};

}  // namespace sovalune
