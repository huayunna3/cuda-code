#pragma once
// ============================================================================
// cuda_bench.cuh —— 各个 CUDA 示例共用的“运行概要”工具
//   CUDA_CHECK             : 统一错误检查（原来散在每个 .cu 里的宏，现在只留一份）
//   cudabench::profile     : 预热 + 多次计时，返回单次平均 kernel 耗时(ms)
//   cudabench::verify_all  : 全量校验结果数组
//   cudabench::Report      : 按统一格式打印概要
// ============================================================================

#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err__) << "\n";        \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)
#endif

namespace cudabench {

// 先预热 warmup_iters 次，再连续计时 profile_iters 次，返回单次平均耗时(ms)
template <typename LaunchFn>
double profile(int warmup_iters, int profile_iters, LaunchFn launch) {
    for (int i = 0; i < warmup_iters; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < profile_iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return profile_iters > 0 ? static_cast<double>(total_ms) / profile_iters : 0.0;
}

// 全量校验：对每个元素调用 check()，全部通过才返回 true
template <typename T, typename CheckFn>
bool verify_all(const std::vector<T>& host, CheckFn check) {
    for (std::size_t i = 0; i < host.size(); ++i) {
        if (!check(host[i])) return false;
    }
    return true;
}

struct Report {
    std::size_t vector_size   = 0;          // 元素个数
    const char* data_type     = "unknown";
    std::size_t element_size  = 0;          // 每个元素的字节数 sizeof(T)
    unsigned    block_threads = 0;
    unsigned    grid_blocks   = 0;
    int         warmup_iters  = 0;
    int         profile_iters = 0;
    double      avg_ms        = 0.0;
    bool        verified      = false;
    double      expected      = 0.0;
    double      got           = 0.0;

    double memory_mb() const {
        return static_cast<double>(vector_size * element_size) / (1024.0 * 1024.0);
    }

    double throughput_gbs() const {
        // vector add 一次要读 a、读 b、写 c，总访存量 = 3 个数组
        double bytes = 3.0 * static_cast<double>(vector_size * element_size);
        return avg_ms > 0.0 ? bytes / (avg_ms * 1.0e6) : 0.0;
    }

    double compute_gflops() const {
        // 每个元素 1 次浮点加法
        double flops = static_cast<double>(vector_size);
        return avg_ms > 0.0 ? flops / (avg_ms * 1.0e6) : 0.0;
    }

    void print(std::ostream& os = std::cout) const {
        const std::ios::fmtflags old_flags = os.flags();
        const std::streamsize    old_prec  = os.precision();

        os << std::left << std::fixed;
        os << std::setw(19) << "Vector size:"     << vector_size << " elements\n";
        os << std::setw(19) << "Data type:"       << data_type << "\n";
        os << std::setprecision(2);
        os << std::setw(19) << "Memory usage:"    << memory_mb() << " MB\n";
        os << std::setprecision(0);
        os << std::setw(19) << "Block size:"      << block_threads << " threads\n";
        os << std::setw(19) << "Grid size:"       << grid_blocks << " blocks\n";
        os << std::setw(19) << "Warm-up iters:"   << warmup_iters << "\n";
        os << std::setw(19) << "Profile iters:"   << profile_iters << "\n";
        os << std::setprecision(6);
        os << std::setw(19) << "Avg kernel time:" << avg_ms << " ms\n";
        os << std::setprecision(2);
        os << std::setw(19) << "Throughput:"      << throughput_gbs() << " GB/s\n";
        os << std::setw(19) << "Compute perf:"    << compute_gflops() << " GFLOP/s\n";
        os << std::setw(19) << "Verification:"    << (verified ? "Passed" : "Failed") << "\n";
        os << std::setw(19) << "Expected:"        << expected << " Got: " << got << "\n";

        os.flags(old_flags);
        os.precision(old_prec);
    }
};

}  // namespace cudabench
