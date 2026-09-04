#include "engine/inference_engine.hpp"
#include "tool_calling/tool_caller.hpp"
#include "nats_bridge/nats_client.hpp"
#include "tokenizer/tokenizer.hpp"
#include <iostream>
#include <signal.h>
#include <thread>
#include <chrono>
#include <cstdlib>

static bool running = true;

void signal_handler(int sig) {
    std::cout << "\n[ModelRuntime] Shutting down..." << std::endl;
    running = false;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Handle --help
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Sovalune Model Runtime v0.1.0" << std::endl;
        std::cout << "Usage: sovalune-model-runtime [OPTIONS]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  -h, --help    Show this help message" << std::endl;
        std::cout << "Environment:" << std::endl;
        std::cout << "  SOVALUNE_MODEL_PATH   Path to model file" << std::endl;
        std::cout << "  SOVALUNE_RUNTIME_MODE cpu or cuda (default: cpu)" << std::endl;
        std::cout << "  SOVALUNE_NATS_URL     NATS server URL" << std::endl;
        return 0;
    }

    std::cout << "=== Sovalune Model Runtime ===" << std::endl;

    // Load config from environment
    sovalune::EngineConfig config;
    config.model_path = std::getenv("SOVALUNE_MODEL_PATH")
        ? std::getenv("SOVALUNE_MODEL_PATH")
        : "";
    config.mode = std::getenv("SOVALUNE_RUNTIME_MODE")
        ? std::getenv("SOVALUNE_RUNTIME_MODE")
        : "cpu";
    config.nats_url = std::getenv("SOVALUNE_NATS_URL")
        ? std::getenv("SOVALUNE_NATS_URL")
        : "nats://localhost:4222";

    // Initialize engine
    sovalune::InferenceEngine engine(config);

    if (!engine.start()) {
        std::cerr << "Failed to start engine" << std::endl;
        return 1;
    }

    std::cout << "Engine ready. Context size: " << engine.get_context_size() << std::endl;
    std::cout << "Model: " << engine.get_model_name() << std::endl;
    std::cout << "Listening for inference requests..." << std::endl;

    // Main loop
    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    engine.stop();
    std::cout << "Model Runtime stopped." << std::endl;
    return 0;
}
