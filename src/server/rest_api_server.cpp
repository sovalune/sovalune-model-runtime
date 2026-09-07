#include "rest_api_server.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#endif

namespace sovalune {

RestApiServer::RestApiServer(const EngineConfig& config, InferenceEngine& engine)
    : config_(config), engine_(engine) {
    port_ = config_.rest_port;
}

RestApiServer::~RestApiServer() {
    stop();
}

bool RestApiServer::start() {
    std::cout << "[REST API] Starting server on port " << port_ << "..." << std::endl;

#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        std::cerr << "[REST API] WSAStartup failed" << std::endl;
        return false;
    }
#endif

    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        std::cerr << "[REST API] Failed to create socket" << std::endl;
        return false;
    }

    // Set socket options
    int opt = 1;
#ifdef _WIN32
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "[REST API] Failed to bind to port " << port_ << std::endl;
#ifdef _WIN32
        closesocket(server_fd_);
#else
        close(server_fd_);
#endif
        return false;
    }

    if (listen(server_fd_, 10) < 0) {
        std::cerr << "[REST API] Failed to listen" << std::endl;
#ifdef _WIN32
        closesocket(server_fd_);
#else
        close(server_fd_);
#endif
        return false;
    }

    running_ = true;
    server_thread_ = std::thread(&RestApiServer::handle_client, this, -1);

    std::cout << "[REST API] Server started on http://0.0.0.0:" << port_ << std::endl;
    std::cout << "[REST API] Endpoints:" << std::endl;
    std::cout << "  GET  /health" << std::endl;
    std::cout << "  GET  /v1/models" << std::endl;
    std::cout << "  POST /v1/chat/completions" << std::endl;
    std::cout << "  POST /v1/completions" << std::endl;

    return true;
}

void RestApiServer::stop() {
    if (!running_) return;

    running_ = false;

#ifdef _WIN32
    if (server_fd_ >= 0) {
        closesocket(server_fd_);
    }
    WSACleanup();
#else
    if (server_fd_ >= 0) {
        close(server_fd_);
    }
#endif

    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    std::cout << "[REST API] Server stopped" << std::endl;
}

bool RestApiServer::is_running() const {
    return running_;
}

int RestApiServer::port() const {
    return port_;
}

void RestApiServer::handle_client(int) {
    while (running_) {
        struct sockaddr_in client_addr;
#ifdef _WIN32
        int client_addr_len = sizeof(client_addr);
#else
        socklen_t client_addr_len = sizeof(client_addr);
#endif

        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_addr_len);
        if (client_fd < 0) {
            if (running_) {
                std::cerr << "[REST API] Accept failed" << std::endl;
            }
            continue;
        }

        // Handle client in separate thread
        std::thread([this, client_fd]() {
            char buffer[4096] = {0};
            int bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);

            if (bytes_read > 0) {
                std::string raw_request(buffer, bytes_read);
                auto request = parse_request(raw_request);
                auto response = route_request(request);
                send_response(client_fd, response);
            }

#ifdef _WIN32
            closesocket(client_fd);
#else
            close(client_fd);
#endif
        }).detach();
    }
}

HttpRequest RestApiServer::parse_request(const std::string& raw_request) {
    HttpRequest request;
    std::istringstream stream(raw_request);
    std::string line;

    // Parse request line
    if (std::getline(stream, line)) {
        std::istringstream line_stream(line);
        line_stream >> request.method >> request.path;
    }

    // Parse headers
    while (std::getline(stream, line) && line != "\r" && !line.empty()) {
        auto colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            // Trim whitespace
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t\r") + 1);
            request.headers[key] = value;
        }
    }

    // Parse body
    std::string remaining;
    while (std::getline(stream, line)) {
        remaining += line + "\n";
    }
    if (!remaining.empty()) {
        request.body = remaining;
    }

    // Parse query parameters
    auto query_pos = request.path.find('?');
    if (query_pos != std::string::npos) {
        std::string query_string = request.path.substr(query_pos + 1);
        request.path = request.path.substr(0, query_pos);

        std::istringstream query_stream(query_string);
        std::string param;
        while (std::getline(query_stream, param, '&')) {
            auto eq_pos = param.find('=');
            if (eq_pos != std::string::npos) {
                request.query_params[param.substr(0, eq_pos)] = param.substr(eq_pos + 1);
            }
        }
    }

    return request;
}

void RestApiServer::send_response(int client_fd, const HttpResponse& response) {
    std::ostringstream stream;
    stream << "HTTP/1.1 " << response.status_code << " " << response.status_text << "\r\n";
    stream << "Content-Type: application/json\r\n";
    stream << "Content-Length: " << response.body.size() << "\r\n";
    stream << "Connection: close\r\n";
    stream << "\r\n";
    stream << response.body;

    std::string raw_response = stream.str();
    send(client_fd, raw_response.c_str(), raw_response.size(), 0);
}

HttpResponse RestApiServer::route_request(const HttpRequest& request) {
    if (request.method == "GET" && request.path == "/health") {
        return handle_health(request);
    } else if (request.method == "GET" && request.path == "/v1/models") {
        return handle_models(request);
    } else if (request.method == "POST" && request.path == "/v1/chat/completions") {
        return handle_chat_completions(request);
    } else if (request.method == "POST" && request.path == "/v1/completions") {
        return handle_completions(request);
    } else {
        HttpResponse response;
        response.status_code = 404;
        response.status_text = "Not Found";
        response.body = "{\"error\": \"Not found\"}";
        return response;
    }
}

HttpResponse RestApiServer::handle_health(const HttpRequest&) {
    HttpResponse response;
    response.status_code = 200;
    response.status_text = "OK";

    std::ostringstream body;
    body << "{"
         << "\"status\": \"" << (engine_.is_ready() ? "ok" : "not ready") << "\","
         << "\"model\": \"" << engine_.get_model_name() << "\","
         << "\"device\": \"" << engine_.get_device_name() << "\","
         << "\"cuda\": " << (engine_.is_cuda() ? "true" : "false") << ","
         << "\"tokens_per_second\": " << engine_.get_tokens_per_second() << ","
         << "\"total_tokens_generated\": " << engine_.get_total_tokens_generated()
         << "}";
    response.body = body.str();

    return response;
}

HttpResponse RestApiServer::handle_models(const HttpRequest&) {
    HttpResponse response;
    response.status_code = 200;
    response.status_text = "OK";

    std::ostringstream body;
    body << "{"
         << "\"object\": \"list\","
         << "\"data\": [{"
         << "\"id\": \"sovalune-model\","
         << "\"object\": \"model\","
         << "\"created\": 1677610602,"
         << "\"owned_by\": \"sovalune\","
         << "\"permission\": [],"
         << "\"root\": \"sovalune-model\","
         << "\"parent\": null"
         << "}]"
         << "}";
    response.body = body.str();

    return response;
}

HttpResponse RestApiServer::handle_chat_completions(const HttpRequest& request) {
    HttpResponse response;

    try {
        auto params = parse_json(request.body);
        std::string model = params.count("model") ? params["model"] : "sovalune-model";
        std::string prompt = params.count("prompt") ? params["prompt"] : "";
        int max_tokens = params.count("max_tokens") ? std::stoi(params["max_tokens"]) : 1024;
        float temperature = params.count("temperature") ? std::stof(params["temperature"]) : 0.7f;

        std::string result = generate_response(model, prompt, max_tokens, temperature);

        std::ostringstream body;
        body << "{"
             << "\"id\": \"chatcmpl-123\","
             << "\"object\": \"chat.completion\","
             << "\"created\": " << std::time(nullptr) << ","
             << "\"model\": \"" << model << "\","
             << "\"usage\": {"
             << "\"prompt_tokens\": " << prompt.size() / 4 << ","
             << "\"completion_tokens\": " << result.size() / 4 << ","
             << "\"total_tokens\": " << (prompt.size() + result.size()) / 4
             << "},"
             << "\"choices\": [{"
             << "\"message\": {"
             << "\"role\": \"assistant\","
             << "\"content\": \"" << result << "\""
             << "},"
             << "\"finish_reason\": \"stop\","
             << "\"index\": 0"
             << "}]"
             << "}";
        response.body = body.str();
        response.status_code = 200;
        response.status_text = "OK";
    } catch (const std::exception& e) {
        response.status_code = 500;
        response.status_text = "Internal Server Error";
        response.body = "{\"error\": \"" + std::string(e.what()) + "\"}";
    }

    return response;
}

HttpResponse RestApiServer::handle_completions(const HttpRequest& request) {
    HttpResponse response;

    try {
        auto params = parse_json(request.body);
        std::string model = params.count("model") ? params["model"] : "sovalune-model";
        std::string prompt = params.count("prompt") ? params["prompt"] : "";
        int max_tokens = params.count("max_tokens") ? std::stoi(params["max_tokens"]) : 1024;
        float temperature = params.count("temperature") ? std::stof(params["temperature"]) : 0.7f;

        std::string result = generate_response(model, prompt, max_tokens, temperature);

        std::ostringstream body;
        body << "{"
             << "\"id\": \"cmpl-123\","
             << "\"object\": \"text_completion\","
             << "\"created\": " << std::time(nullptr) << ","
             << "\"model\": \"" << model << "\","
             << "\"choices\": [{"
             << "\"text\": \"" << result << "\","
             << "\"index\": 0,"
             << "\"logprobs\": null,"
             << "\"finish_reason\": \"stop\""
             << "}],"
             << "\"usage\": {"
             << "\"prompt_tokens\": " << prompt.size() / 4 << ","
             << "\"completion_tokens\": " << result.size() / 4 << ","
             << "\"total_tokens\": " << (prompt.size() + result.size()) / 4
             << "}"
             << "}";
        response.body = body.str();
        response.status_code = 200;
        response.status_text = "OK";
    } catch (const std::exception& e) {
        response.status_code = 500;
        response.status_text = "Internal Server Error";
        response.body = "{\"error\": \"" + std::string(e.what()) + "\"}";
    }

    return response;
}

std::unordered_map<std::string, std::string> RestApiServer::parse_json(const std::string& json) {
    std::unordered_map<std::string, std::string> result;

    // Simple JSON parser for flat objects
    size_t pos = 0;
    while (pos < json.size()) {
        // Find key
        auto key_start = json.find('"', pos);
        if (key_start == std::string::npos) break;
        auto key_end = json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = json.substr(key_start + 1, key_end - key_start - 1);

        // Find value
        auto colon_pos = json.find(':', key_end + 1);
        if (colon_pos == std::string::npos) break;

        auto value_start = json.find_first_not_of(" \t\n\r", colon_pos + 1);
        if (value_start == std::string::npos) break;

        std::string value;
        if (json[value_start] == '"') {
            // String value
            auto value_end = json.find('"', value_start + 1);
            if (value_end == std::string::npos) break;
            value = json.substr(value_start + 1, value_end - value_start - 1);
            pos = value_end + 1;
        } else {
            // Number or other value
            auto value_end = json.find_first_of(",}", value_start);
            if (value_end == std::string::npos) break;
            value = json.substr(value_start, value_end - value_start);
            pos = value_end;
        }

        result[key] = value;

        // Skip comma
        if (pos < json.size() && json[pos] == ',') {
            pos++;
        }
    }

    return result;
}

std::string RestApiServer::generate_response(
    const std::string& model,
    const std::string& prompt,
    int max_tokens,
    float temperature
) {
    GenerationConfig gen_config;
    gen_config.max_tokens = max_tokens;
    gen_config.temperature = temperature;

    return engine_.generate(prompt, gen_config);
}

}  // namespace sovalune
