#pragma once

#include <string>
#include <functional>
#include <unordered_map>

namespace sovalune {

struct InferenceMessage {
    std::string subject;
    std::string reply;
    std::string payload;
};

using MessageHandler = std::function<void(const InferenceMessage&)>;

class MessageHandlerDispatcher {
public:
    void register_handler(const std::string& subject, MessageHandler handler);
    void dispatch(const InferenceMessage& message);

private:
    std::unordered_map<std::string, MessageHandler> handlers_;
};

}  // namespace sovalune
