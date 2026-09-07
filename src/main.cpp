#include "engine/inference_engine.hpp"
#include "engine/gpu_manager.hpp"
#include "server/rest_api_server.hpp"
#include "tool_calling/tool_caller.hpp"
#include "nats_bridge/nats_client.hpp"
#include "tokenizer/tokenizer.hpp"
#include <iostream>
#include <signal.h>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <cstring>

static bool running = true;

void signal_handler(int sig) {
    std::cout << "\n[ModelRuntime] Shutting down (signal " << sig << ")..." << std::endl;
    running = false;
}

void print_usage() {
    std::cout << "Sovalune Model Runtime v0.2.0" << std::endl;
    std::cout << "C++/CUDA inference engine for Sovalune AI models" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: sovalune-model-runtime [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << "  -v, --version           Show version" << std::endl;
    std::cout << "  -m, --model PATH        Path to GGUF model file" << std::endl;
    std::cout << "  -c, --context N         Context size (default: 4096)" << std::endl;
    std::cout << "  -t, --threads N         CPU threads (default: 4)" << std::endl;
    std::cout << "  --gpu-layers N          GPU layers (default: 99)" << std::endl;
    std::cout << "  --gpu-device N          GPU device ID (default: auto)" << std::endl;
    std::cout << "  --port N                REST API port (default: 8080)" << std::endl;
    std::cout << "  --no-nats               Disable NATS" << std::endl;
    std::cout << "  --verbose               Verbose output" << std::endl;
    std::cout << std::endl;
    std::cout << "Environment Variables:" << std::endl;
    std::cout << "  SOVALUNE_MODEL_PATH     Path to GGUF model file" << std::endl;
    std::cout << "  SOVALUNE_RUNTIME_MODE   cpu or cuda (default: cpu)" << std::endl;
    std::cout << "  SOVALUNE_NATS_URL       NATS server URL" << std::endl;
    std::cout << "  SOVALUNE_REST_PORT      REST API port" << std::endl;
}

void print_version() {
    std::cout << "sovalune-model-runtime 0.2.0" << std::endl;
    std::cout << "Build: " << __DATE__ << " " << __TIME__ << std::endl;
#ifdef SOVALUNE_CUDA_ENABLED
    std::cout << "CUDA: enabled" << std::endl;
#else
    std::cout << "CUDA: disabled" << std::endl;
#endif
#ifdef SOVALUNE_HAS_LLAMA_CPP
    std::cout << "llama.cpp: enabled" << std::endl;
#else
    std::cout << "llama.cpp: disabled" << std::endl;
#endif
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    sovalune::EngineConfig config;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "-v" || arg == "--version") {
            print_version();
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                config.model_path = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires an argument" << std::endl;
                return 1;
            }
        } else if (arg == "-c" || arg == "--context") {
            if (i + 1 < argc) {
                config.n_ctx = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: " << arg << " requires an argument" << std::endl;
                return 1;
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                config.n_threads = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: " << arg << " requires an argument" << std::endl;
                return 1;
            }
        } else if (arg == "--gpu-layers") {
            if (i + 1 < argc) {
                config.n_gpu_layers = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: " << arg << " requires an argument" << std::endl;
                return 1;
            }
        } else if (arg == "--gpu-device") {
            if (i + 1 < argc) {
                config.gpu_device = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: " << arg << " requires an argument" << std::endl;
                return 1;
            }
        } else if (arg == "--port") {
            if (i + 1 < argc) {
                config.rest_port = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: " << arg << " requires an argument" << std::endl;
                return 1;
            }
        } else if (arg == "--no-nats") {
            config.nats_url.clear();
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }

    // Override with environment variables
    if (config.model_path.empty() && std::getenv("SOVALUNE_MODEL_PATH")) {
        config.model_path = std::getenv("SOVALUNE_MODEL_PATH");
    }
    if (std::getenv("SOVALUNE_RUNTIME_MODE")) {
        config.mode = std::getenv("SOVALUNE_RUNTIME_MODE");
    }
    if (config.nats_url.empty() && std::getenv("SOVALUNE_NATS_URL")) {
        config.nats_url = std::getenv("SOVALUNE_NATS_URL");
    }
    if (std::getenv("SOVALUNE_REST_PORT")) {
        config.rest_port = std::stoi(std::getenv("SOVALUNE_REST_PORT"));
    }

    std::cout << "=== Sovalune Model Runtime ===" << std::endl;
    print_version();
    std::cout << std::endl;

    // Initialize engine
    sovalune::InferenceEngine engine(config);

    if (!engine.start()) {
        std::cerr << "Failed to start engine" << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "Engine ready." << std::endl;
    std::cout << "  Context size: " << engine.get_context_size() << std::endl;
    std::cout << "  Model: " << engine.get_model_name() << std::endl;
    std::cout << "  Device: " << engine.get_device_name() << std::endl;
    std::cout << std::endl;

    // Start REST API server
    sovalune::RestApiServer server(config, engine);
    if (!server.start()) {
        std::cerr << "Failed to start REST API server" << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "Listening for inference requests..." << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;
    std::cout << std::endl;

    // Main loop
    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Print stats periodically
        static int stats_counter = 0;
        if (++stats_counter >= 50) { // Every 5 seconds
            stats_counter = 0;
            double tps = engine.get_tokens_per_second();
            if (tps > 0) {
                std::cout << "[Stats] Tokens/sec: " << tps
                          << " | Total: " << engine.get_total_tokens_generated() << std::endl;
            }
        }
    }

    // Shutdown
    std::cout << std::endl;
    std::cout << "Shutting down..." << std::endl;

    server.stop();
    engine.stop();

    std::cout << "Model Runtime stopped." << std::endl;
    return 0;
}
