#pragma once

#include "../engine/engine_config.hpp"
#include "../engine/inference_engine.hpp"
#include <string>
#include <functional>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace sovalune {

/// HTTP request
struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::unordered_map<std::string, std::string> headers;
    std::unordered_map<std::string, std::string> query_params;
};

/// HTTP response
struct HttpResponse {
    int status_code = 200;
    std::string status_text = "OK";
    std::string body;
    std::unordered_map<std::string, std::string> headers;
};

/// REST API server for inference requests
class RestApiServer {
public:
    RestApiServer(const EngineConfig& config, InferenceEngine& engine);
    ~RestApiServer();

    RestApiServer(const RestApiServer&) = delete;
    RestApiServer& operator=(const RestApiServer&) = delete;

    /// Start the server
    bool start();

    /// Stop the server
    void stop();

    /// Check if server is running
    bool is_running() const;

    /// Get server port
    int port() const;

private:
    EngineConfig config_;
    InferenceEngine& engine_;
    std::atomic<bool> running_{false};
    std::thread server_thread_;
    int port_ = 8080;
    int server_fd_ = -1;

    /// Handle a client connection
    void handle_client(int client_fd);

    /// Parse HTTP request
    HttpRequest parse_request(const std::string& raw_request);

    /// Send HTTP response
    void send_response(int client_fd, const HttpResponse& response);

    /// Route request to handler
    HttpResponse route_request(const HttpRequest& request);

    /// API handlers
    HttpResponse handle_health(const HttpRequest& request);
    HttpResponse handle_models(const HttpRequest& request);
    HttpResponse handle_chat_completions(const HttpRequest& request);
    HttpResponse handle_completions(const HttpRequest& request);

    /// Parse JSON body
    std::unordered_map<std::string, std::string> parse_json(const std::string& json);

    /// Generate response for API
    std::string generate_response(const std::string& model, const std::string& prompt, int max_tokens, float temperature);
};

}  // namespace sovalune
