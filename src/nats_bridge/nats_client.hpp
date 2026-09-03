#pragma once

#include <string>
#include <vector>
#include <nats.h>

namespace sovalune {

struct InferenceRequest {
    std::string request_id;
    std::string session_id;
    std::string prompt_context;
    int max_tokens = 1024;
    float temperature = 0.7f;
};

struct InferenceResponse {
    std::string request_id;
    std::string delta;
    bool done = false;
    std::string message_id;
};

using ResponseCallback = std::function<void(const InferenceResponse&)>;

class NatsClient {
public:
    explicit NatsClient(const std::string& url);
    ~NatsClient();
    
    // Subscribe to inference requests
    void subscribe_inference(ResponseCallback callback);
    
    // Publish inference response
    void publish_response(const InferenceResponse& response);
    
    // Publish tool call
    void publish_tool_call(const std::string& request_id, const std::string& tool_name, const std::string& arguments);
    
    // Wait for tool result
    std::string wait_for_tool_result(const std::string& request_id, int timeout_ms = 30000);
    
    // Health check
    bool is_connected() const;
    
private:
    natsConnection* conn_ = nullptr;
    natsSubscription* sub_ = nullptr;
};

}  // namespace sovalune
