#include "gpu_manager.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>

#ifdef SOVALUNE_CUDA_ENABLED
#include <cuda_runtime.h>
#include <cuda.h>
#endif

namespace sovalune {

#ifdef SOVALUNE_CUDA_ENABLED

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "[GPU] CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    } \
} while(0)

// CUDA memory pool implementation
struct GpuMemoryPool::Impl {
    int device_id;
    void* base_ptr = nullptr;
    size_t total_size = 0;
    size_t used_size = 0;
    size_t reserved_size = 0;

    explicit GpuMemoryPool(int dev_id) : device_id(dev_id) {
        // Pre-allocate 256MB pool
        total_size = 256 * 1024 * 1024;
        GpuDeviceGuard guard(device_id);
        cudaError_t err = cudaMalloc(&base_ptr, total_size);
        if (err == cudaSuccess) {
            reserved_size = total_size;
            std::cout << "[GPU] Memory pool allocated: " << (total_size / 1024 / 1024) << " MB" << std::endl;
        } else {
            std::cerr << "[GPU] Failed to allocate memory pool: " << cudaGetErrorString(err) << std::endl;
            base_ptr = nullptr;
            total_size = 0;
        }
    }

    ~Impl() {
        if (base_ptr) {
            GpuDeviceGuard guard(device_id);
            cudaFree(base_ptr);
        }
    }
};

GpuMemoryPool::GpuMemoryPool(int device_id)
    : impl_(std::make_unique<Impl>(device_id)) {}

GpuMemoryPool::~GpuMemoryPool() = default;

void* GpuMemoryPool::allocate(size_t size) {
    if (!impl_->base_ptr) return nullptr;

    // Simple bump allocator
    size_t aligned_size = (size + 255) & ~255; // Align to 256 bytes
    if (impl_->used_size + aligned_size > impl_->total_size) {
        return nullptr; // Out of memory
    }

    void* ptr = static_cast<char*>(impl_->base_ptr) + impl_->used_size;
    impl_->used_size += aligned_size;
    return ptr;
}

void GpuMemoryPool::deallocate(void* ptr) {
    // Pool is reset all at once
    (void)ptr;
}

size_t GpuMemoryPool::total_allocated() const {
    return impl_->total_size;
}

size_t GpuMemoryPool::total_reserved() const {
    return impl_->reserved_size;
}

size_t GpuMemoryPool::total_used() const {
    return impl_->used_size;
}

void GpuMemoryPool::reset() {
    impl_->used_size = 0;
}

void GpuMemoryPool::trim(size_t max_cached) {
    (void)max_cached;
    // No-op for now
}

// GpuManager implementation
bool GpuManager::initialized_ = false;
std::vector<GpuDevice> GpuManager::devices_;
std::vector<std::unique_ptr<GpuMemoryPool>> GpuManager::memory_pools_;

bool GpuManager::initialize() {
    if (initialized_) return true;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::cerr << "[GPU] Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    if (device_count == 0) {
        std::cerr << "[GPU] No CUDA devices found" << std::endl;
        return false;
    }

    std::cout << "[GPU] Found " << device_count << " CUDA device(s)" << std::endl;

    devices_.resize(device_count);
    memory_pools_.resize(device_count);

    for (int i = 0; i < device_count; ++i) {
        devices_[i] = get_device_info(i);
        memory_pools_[i] = std::make_unique<GpuMemoryPool>(i);
        print_device_info(i);
    }

    initialized_ = true;
    return true;
}

void GpuManager::shutdown() {
    memory_pools_.clear();
    devices_.clear();
    initialized_ = false;
}

int GpuManager::device_count() {
    if (!initialized_) return 0;
    return static_cast<int>(devices_.size());
}

GpuDevice GpuManager::get_device_info(int device_id) {
    GpuDevice info{};
    info.device_id = device_id;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        return info;
    }

    info.name = prop.name;
    info.total_memory = prop.totalGlobalMem;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_shared_memory_per_block = prop.sharedMemPerBlock;
    info.supports_fp16 = (prop.major >= 5); // SM 5.0+ supports FP16
    info.supports_int8 = (prop.major >= 6); // SM 6.0+ supports INT8

    // Get free memory
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    info.free_memory = static_cast<int64_t>(free_mem);
    info.used_memory = info.total_memory - info.free_memory;

    return info;
}

int GpuManager::select_best_device() {
    if (!initialized_ || devices_.empty()) return -1;

    int best_device = 0;
    int64_t max_free_memory = 0;

    for (const auto& device : devices_) {
        if (device.free_memory > max_free_memory) {
            max_free_memory = device.free_memory;
            best_device = device.device_id;
        }
    }

    return best_device;
}

bool GpuManager::set_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    return err == cudaSuccess;
}

int GpuManager::current_device() {
    int device = -1;
    cudaGetDevice(&device);
    return device;
}

std::string GpuManager::device_name(int device_id) {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return "unknown";
    }
    return devices_[device_id].name;
}

size_t GpuManager::get_device_total_memory(int device_id) {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return 0;
    }
    return static_cast<size_t>(devices_[device_id].total_memory);
}

size_t GpuManager::get_device_free_memory(int device_id) {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return 0;
    }
    return static_cast<size_t>(devices_[device_id].free_memory);
}

bool GpuManager::is_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

bool GpuManager::check_device_compatibility(int device_id, bool need_fp16, bool need_int8) {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return false;
    }

    const auto& device = devices_[device_id];

    if (need_fp16 && !device.supports_fp16) return false;
    if (need_int8 && !device.supports_int8) return false;

    return true;
}

void GpuManager::print_device_info(int device_id) {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return;
    }

    const auto& device = devices_[device_id];
    std::cout << "[GPU] Device " << device_id << ": " << device.name << std::endl;
    std::cout << "[GPU]   Compute capability: " << device.compute_capability_major
              << "." << device.compute_capability_minor << std::endl;
    std::cout << "[GPU]   Total memory: " << (device.total_memory / 1024 / 1024) << " MB" << std::endl;
    std::cout << "[GPU]   Free memory: " << (device.free_memory / 1024 / 1024) << " MB" << std::endl;
    std::cout << "[GPU]   Multiprocessors: " << device.multiprocessor_count << std::endl;
    std::cout << "[GPU]   FP16: " << (device.supports_fp16 ? "yes" : "no") << std::endl;
    std::cout << "[GPU]   INT8: " << (device.supports_int8 ? "yes" : "no") << std::endl;
}

void GpuManager::print_all_devices() {
    for (size_t i = 0; i < devices_.size(); ++i) {
        print_device_info(static_cast<int>(i));
    }
}

GpuMemoryPool& GpuManager::get_memory_pool(int device_id) {
    if (device_id < 0 || device_id >= static_cast<int>(memory_pools_.size())) {
        throw std::runtime_error("Invalid device ID");
    }
    return *memory_pools_[device_id];
}

// RAII device guard
GpuDeviceGuard::GpuDeviceGuard(int device_id)
    : device_id_(device_id), previous_device_(-1) {
    cudaGetDevice(&previous_device_);
    if (previous_device_ != device_id_) {
        cudaSetDevice(device_id_);
    }
}

GpuDeviceGuard::~GpuDeviceGuard() {
    if (previous_device_ != device_id_) {
        cudaSetDevice(previous_device_);
    }
}

#else // No CUDA support

// Stub implementations when CUDA is not available
bool GpuManager::initialize() {
    std::cout << "[GPU] CUDA not available, running in CPU mode" << std::endl;
    return true;
}

void GpuManager::shutdown() {}

int GpuManager::device_count() { return 0; }

GpuDevice GpuManager::get_device_info(int device_id) {
    return {};
}

int GpuManager::select_best_device() { return -1; }

bool GpuManager::set_device(int device_id) { return false; }

int GpuManager::current_device() { return -1; }

std::string GpuManager::device_name(int device_id) { return "cpu"; }

size_t GpuManager::get_device_total_memory(int device_id) { return 0; }

size_t GpuManager::get_device_free_memory(int device_id) { return 0; }

bool GpuManager::is_available() { return false; }

bool GpuManager::check_device_compatibility(int device_id, bool need_fp16, bool need_int8) { return false; }

void GpuManager::print_device_info(int device_id) {
    std::cout << "[GPU] CUDA not available" << std::endl;
}

void GpuManager::print_all_devices() {
    std::cout << "[GPU] CUDA not available" << std::endl;
}

GpuMemoryPool& GpuManager::get_memory_pool(int device_id) {
    static GpuMemoryPool stub_pool(-1);
    return stub_pool;
}

// GpuMemoryPool stub
struct GpuMemoryPool::Impl {
    int device_id;
    explicit GpuMemoryPool(int dev_id) : device_id(dev_id) {}
};

GpuMemoryPool::GpuMemoryPool(int device_id)
    : impl_(std::make_unique<Impl>(device_id)) {}

GpuMemoryPool::~GpuMemoryPool() = default;

void* GpuMemoryPool::allocate(size_t size) { return nullptr; }
void GpuMemoryPool::deallocate(void* ptr) {}
size_t GpuMemoryPool::total_allocated() const { return 0; }
size_t GpuMemoryPool::total_reserved() const { return 0; }
size_t GpuMemoryPool::total_used() const { return 0; }
void GpuMemoryPool::reset() {}
void GpuMemoryPool::trim(size_t max_cached) {}

// GpuDeviceGuard stub
GpuDeviceGuard::GpuDeviceGuard(int device_id) : device_id_(device_id), previous_device_(-1) {}
GpuDeviceGuard::~GpuDeviceGuard() = default;

#endif // SOVALUNE_CUDA_ENABLED

}  // namespace sovalune
