#include "nats_client.hpp"
#include <iostream>

namespace sovalune {

NatsClient::NatsClient(const std::string& url) {
    natsStatus s = natsConnection_Connect(&conn_, url.c_str());
    if (s == NATS_OK) {
        std::cout << "[NATS] Connected to " << url << std::endl;
    } else {
        std::cerr << "[NATS] Failed to connect: " << natsStatus_GetText(s) << std::endl;
    }
}

NatsClient::~NatsClient() {
    if (sub_) {
        natsSubscription_Destroy(sub_);
    }
    if (conn_) {
        natsConnection_Destroy(conn_);
    }
}

void NatsClient::subscribe_inference(ResponseCallback callback) {
    // TODO: Implement NATS subscription
    std::cout << "[NATS] Subscribed to inference.requests" << std::endl;
}

void NatsClient::publish_response(const InferenceResponse& response) {
    // TODO: Implement NATS publish
    std::cout << "[NATS] Published response for " << response.request_id << std::endl;
}

void NatsClient::publish_tool_call(const std::string& request_id, const std::string& tool_name, const std::string& arguments) {
    // TODO: Implement NATS publish
    std::cout << "[NATS] Published tool call: " << tool_name << std::endl;
}

std::string NatsClient::wait_for_tool_result(const std::string& request_id, int timeout_ms) {
    // TODO: Implement NATS request/reply
    return "{}";
}

bool NatsClient::is_connected() const {
    return conn_ != nullptr;
}

}  // namespace sovalune
