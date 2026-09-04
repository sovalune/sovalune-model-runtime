#include "nats_client.hpp"
#include <iostream>
#include <thread>
#include <chrono>

namespace sovalune {

NatsClient::NatsClient(const std::string& url) : url_(url) {
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
    if (last_msg_) {
        natsMsg_Destroy(last_msg_);
    }
    if (sub_) {
        natsSubscription_Destroy(sub_);
    }
    if (conn_) {
        natsConnection_Destroy(conn_);
    }
#endif
}

void NatsClient::subscribe_inference(ResponseCallback callback) {
    callback_ = callback;
#if SOVALUNE_HAS_NATS
    if (!conn_) return;

    natsStatus s = natsSubscriptionNextMsg(&sub_, conn_, "inference.requests", 1000000000LL);
    if (s == NATS_OK) {
        std::cout << "[NATS] Subscribed to inference.requests" << std::endl;
    } else {
        std::cerr << "[NATS] Failed to subscribe: " << natsStatus_GetText(s) << std::endl;
    }
#else
    (void)callback;
    std::cout << "[NATS] Subscribed (stub)" << std::endl;
#endif
}

void NatsClient::publish_response(const InferenceResponse& response) {
#if SOVALUNE_HAS_NATS
    if (!conn_) return;

    // Serialize response to JSON
    std::string json = "{\"request_id\":\"" + response.request_id +
                       "\",\"delta\":\"" + response.delta +
                       "\",\"done\":" + (response.done ? "true" : "false") +
                       ",\"message_id\":\"" + response.message_id + "\"}";

    std::string subject = "inference.responses." + response.request_id;
    natsStatus s = natsConnection_Publish(conn_, subject.c_str(), json.c_str(), json.size());
    if (s != NATS_OK) {
        std::cerr << "[NATS] Failed to publish: " << natsStatus_GetText(s) << std::endl;
    }

    // Also publish to completions topic
    if (response.done) {
        natsConnection_Publish(conn_, "inference.completed", json.c_str(), json.size());
    }
#else
    (void)response;
    std::cout << "[NATS] Published response (stub)" << std::endl;
#endif
}

void NatsClient::publish_tool_call(const std::string& request_id, const std::string& tool_name, const std::string& arguments) {
#if SOVALUNE_HAS_NATS
    if (!conn_) return;

    std::string json = "{\"request_id\":\"" + request_id +
                       "\",\"tool\":\"" + tool_name +
                       "\",\"arguments\":" + arguments + "}";

    std::string subject = "inference.tool_calls." + tool_name;
    natsConnection_Publish(conn_, subject.c_str(), json.c_str(), json.size());
#else
    (void)request_id;
    (void)tool_name;
    (void)arguments;
    std::cout << "[NATS] Published tool call (stub)" << std::endl;
#endif
}

std::string NatsClient::wait_for_tool_result(const std::string& request_id, int timeout_ms) {
#if SOVALUNE_HAS_NATS
    if (!conn_) return "{}";

    std::string reply_subject = "inference.tool_results." + request_id;
    natsSubscription* sub = nullptr;
    natsMsg* msg = nullptr;

    natsStatus s = natsConnection_Request(&msg, conn_, reply_subject.c_str(), nullptr, 0, timeout_ms);
    if (s == NATS_OK && msg) {
        std::string result(natsMsg_GetData(msg), natsMsg_GetDataLength(msg));
        natsMsg_Destroy(msg);
        return result;
    }
    return "{}";
#else
    (void)request_id;
    (void)timeout_ms;
    return "{}";
#endif
}

bool NatsClient::is_connected() const {
#if SOVALUNE_HAS_NATS
    return conn_ != nullptr;
#else
    return false;
#endif
}

}  // namespace sovalune
