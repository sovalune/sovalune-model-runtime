#include "nats_client.hpp"
#include <iostream>

namespace sovalune {

NatsClient::NatsClient(const std::string& url) {
#if SOVALUNE_HAS_NATS
    natsStatus s = natsConnection_Connect(&conn_, url.c_str());
    if (s == NATS_OK) {
        std::cout << "[NATS] Connected to " << url << std::endl;
    } else {
        std::cerr << "[NATS] Failed to connect: " << natsStatus_GetText(s) << std::endl;
    }
#else
    (void)url;
    std::cout << "[NATS] Compiled without NATS support" << std::endl;
#endif
}

NatsClient::~NatsClient() {
#if SOVALUNE_HAS_NATS
    if (sub_) {
        natsSubscription_Destroy(sub_);
    }
    if (conn_) {
        natsConnection_Destroy(conn_);
    }
#endif
}

void NatsClient::subscribe_inference(ResponseCallback callback) {
    (void)callback;
    std::cout << "[NATS] Subscribed to inference.requests" << std::endl;
}

void NatsClient::publish_response(const InferenceResponse& response) {
    std::cout << "[NATS] Published response for " << response.request_id << std::endl;
}

void NatsClient::publish_tool_call(const std::string& request_id, const std::string& tool_name, const std::string& arguments) {
    (void)request_id;
    (void)arguments;
    std::cout << "[NATS] Published tool call: " << tool_name << std::endl;
}

std::string NatsClient::wait_for_tool_result(const std::string& request_id, int timeout_ms) {
    (void)request_id;
    (void)timeout_ms;
    return "{}";
}

bool NatsClient::is_connected() const {
#if SOVALUNE_HAS_NATS
    return conn_ != nullptr;
#else
    return false;
#endif
}

}  // namespace sovalune
