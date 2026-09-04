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
    void register_handler(const std::string& subject, MessageHandler handler) {
        handlers_[subject] = handler;
    }

    void dispatch(const InferenceMessage& message) {
        auto it = handlers_.find(message.subject);
        if (it != handlers_.end()) {
            it->second(message);
        }
    }

private:
    std::unordered_map<std::string, MessageHandler> handlers_;
};

}  // namespace sovalune
