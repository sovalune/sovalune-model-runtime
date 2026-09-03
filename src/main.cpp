#include "engine/inference_engine.hpp"
#include "tool_calling/tool_caller.hpp"
#include "nats_bridge/nats_client.hpp"
#include "tokenizer/tokenizer.hpp"
#include <iostream>
#include <signal.h>

static bool running = true;

void signal_handler(int sig) {
    std::cout << "\n[ModelRuntime] Shutting down..." << std::endl;
    running = false;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "=== Sovalune Model Runtime ===" << std::endl;
    
    // Load config
    sovalune::EngineConfig config;
    config.model_path = std::getenv("SOVALUNE_MODEL_PATH") 
        ? std::getenv("SOVALUNE_MODEL_PATH") 
        : "";
    config.mode = std::getenv("SOVALUNE_RUNTIME_MODE") 
        ? std::getenv("SOVALUNE_RUNTIME_MODE") 
        : "cpu";
    
    std::cout << "Mode: " << config.mode << std::endl;
    std::cout << "Model: " << (config.model_path.empty() ? "(none)" : config.model_path) << std::endl;
    
    // Initialize engine
    sovalune::InferenceEngine engine(config);
    
    if (!engine.is_ready()) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return 1;
    }
    
    std::cout << "Engine ready. Context size: " << engine.get_context_size() << std::endl;
    
    // Connect to NATS
    std::string nats_url = std::getenv("SOVALUNE_NATS_URL") 
        ? std::getenv("SOVALUNE_NATS_URL") 
        : "nats://localhost:4222";
    
    sovalune::NatsClient nats(nats_url);
    
    if (!nats.is_connected()) {
        std::cerr << "Failed to connect to NATS" << std::endl;
        return 1;
    }
    
    // Register tools
    sovalune::ToolCaller tool_caller;
    // TODO: Register memory_search, memory_write, etc.
    
    std::cout << "Listening for inference requests..." << std::endl;
    
    // Main loop
    while (running) {
        // TODO: Process NATS messages
        // For now, just sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "Model Runtime stopped." << std::endl;
    return 0;
}
