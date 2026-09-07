#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace sovalune {

/// GPU device information
struct GpuDevice {
    int device_id;
    std::string name;
    int64_t total_memory;      // bytes
    int64_t free_memory;       // bytes
    int64_t used_memory;       // bytes
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_shared_memory_per_block;
    bool supports_fp16;
    bool supports_fp32;
    bool supports_int8;
};

/// GPU memory pool for efficient allocation
class GpuMemoryPool {
public:
    GpuMemoryPool(int device_id);
    ~GpuMemoryPool();

    GpuMemoryPool(const GpuMemoryPool&) = delete;
    GpuMemoryPool& operator=(const GpuMemoryPool&) = delete;

    /// Allocate GPU memory
    void* allocate(size_t size);

    /// Free GPU memory
    void deallocate(void* ptr);

    /// Get memory stats
    size_t total_allocated() const;
    size_t total_reserved() const;
    size_t total_used() const;

    /// Reset pool (free all cached memory)
    void reset();

    /// Trim memory pool to release unused memory
    void trim(size_t max_cached = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// CUDA GPU manager for device selection and memory management
class GpuManager {
public:
    GpuManager() = default;
    ~GpuManager() = default;

    /// Initialize CUDA and enumerate devices
    static bool initialize();

    /// Shutdown CUDA
    static void shutdown();

    /// Get number of available GPUs
    static int device_count();

    /// Get information about a specific GPU
    static GpuDevice get_device_info(int device_id);

    /// Get best device for inference (most free memory)
    static int select_best_device();

    /// Set current CUDA device
    static bool set_device(int device_id);

    /// Get current CUDA device
    static int current_device();

    /// Get device name
    static std::string device_name(int device_id);

    /// Get device memory info
    static size_t get_device_total_memory(int device_id);
    static size_t get_device_free_memory(int device_id);

    /// Check if CUDA is available
    static bool is_available();

    /// Check if device supports required features
    static bool check_device_compatibility(int device_id, bool need_fp16 = false, bool need_int8 = false);

    /// Print device info
    static void print_device_info(int device_id);

    /// Print all devices
    static void print_all_devices();

    /// Get memory pool for a device
    static GpuMemoryPool& get_memory_pool(int device_id);

private:
    static bool initialized_;
    static std::vector<GpuDevice> devices_;
    static std::vector<std::unique_ptr<GpuMemoryPool>> memory_pools_;
};

/// RAII device selector
class GpuDeviceGuard {
public:
    explicit GpuDeviceGuard(int device_id);
    ~GpuDeviceGuard();

    GpuDeviceGuard(const GpuDeviceGuard&) = delete;
    GpuDeviceGuard& operator=(const GpuDeviceGuard&) = delete;

    int device_id() const { return device_id_; }

private:
    int device_id_;
    int previous_device_;
};

}  // namespace sovalune
