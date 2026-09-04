#include "message_handler.hpp"
#include <iostream>

namespace sovalune {

void MessageHandlerDispatcher::register_handler(const std::string& subject, MessageHandler handler) {
    handlers_[subject] = handler;
}

void MessageHandlerDispatcher::dispatch(const InferenceMessage& message) {
    auto it = handlers_.find(message.subject);
    if (it != handlers_.end()) {
        it->second(message);
    } else {
        std::cerr << "[MessageHandler] No handler for subject: " << message.subject << std::endl;
    }
}

}  // namespace sovalune
