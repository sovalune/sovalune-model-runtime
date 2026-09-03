#include "engine_config.hpp"

namespace sovalune {

// Default configuration
EngineConfig default_config() {
    EngineConfig config;
    config.model_path = std::getenv("SOVALUNE_MODEL_PATH") 
        ? std::getenv("SOVALUNE_MODEL_PATH") 
        : "";
    config.mode = std::getenv("SOVALUNE_RUNTIME_MODE") 
        ? std::getenv("SOVALUNE_RUNTIME_MODE") 
        : "cpu";
    return config;
}

}  // namespace sovalune
