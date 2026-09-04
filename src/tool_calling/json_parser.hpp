#pragma once

#include <string>
#include <optional>
#include <sstream>

namespace sovalune {

struct ParsedToolCall {
    std::string name;
    std::string arguments;
};

class JsonParser {
public:
    static std::optional<ParsedToolCall> parse_tool_call(const std::string& json_str) {
        auto name_pos = json_str.find("\"name\"");
        if (name_pos == std::string::npos) {
            name_pos = json_str.find("\"tool\"");
        }
        if (name_pos == std::string::npos) return std::nullopt;

        auto colon_pos = json_str.find(':', name_pos);
        if (colon_pos == std::string::npos) return std::nullopt;

        auto quote_start = json_str.find('"', colon_pos + 1);
        if (quote_start == std::string::npos) return std::nullopt;

        auto quote_end = json_str.find('"', quote_start + 1);
        if (quote_end == std::string::npos) return std::nullopt;

        ParsedToolCall result;
        result.name = json_str.substr(quote_start + 1, quote_end - quote_start - 1);

        auto args_pos = json_str.find("\"arguments\"");
        if (args_pos == std::string::npos) {
            args_pos = json_str.find("\"parameters\"");
        }
        if (args_pos != std::string::npos) {
            auto brace_pos = json_str.find('{', args_pos);
            if (brace_pos != std::string::npos) {
                int depth = 0;
                size_t end = brace_pos;
                for (size_t i = brace_pos; i < json_str.size(); ++i) {
                    if (json_str[i] == '{') depth++;
                    else if (json_str[i] == '}') {
                        depth--;
                        if (depth == 0) { end = i + 1; break; }
                    }
                }
                result.arguments = json_str.substr(brace_pos, end - brace_pos);
            }
        }

        return result;
    }

    static std::string serialize_response(const std::string& request_id, const std::string& content, bool done) {
        std::ostringstream ss;
        ss << "{\"request_id\":\"" << request_id
           << "\",\"delta\":\"" << content
           << "\",\"done\":" << (done ? "true" : "false")
           << "}";
        return ss.str();
    }

    static std::string serialize_tool_result(const std::string& request_id, const std::string& tool_name, const std::string& result) {
        std::ostringstream ss;
        ss << "{\"request_id\":\"" << request_id
           << "\",\"tool_name\":\"" << tool_name
           << "\",\"result\":" << result
           << "}";
        return ss.str();
    }
};

}  // namespace sovalune
