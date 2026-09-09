#include <iostream>
#include <vector>
#include <cstdlib>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 步骤5（底部）：错误检查宏 + 运行概要工具（见 common/cuda_bench.cuh）
#include "../common/cuda_bench.cuh"

// CUDA 只内置 half / half2，没有 half4：仿照 float4 自定义（8 字节、8 字节对齐）
struct __align__(8) half4 {
    half x, y, z, w;
};

// 仿 make_float4 提供 make_half4（host/device 通用）
__device__ __host__ half4 make_half4(half x, half y, half z, half w) {
    half4 r;
    r.x = x; r.y = y; r.z = z; r.w = w;
    return r;
}

// 图2：__device__ 模板 add() —— 统一封装不同类型加法（仅在 GPU 上运行）
// half：CUDA 内置半精度类型（2 字节：1 符号 + 5 指数 + 10 尾数）
template <typename T>
__device__ T add(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, float>) {
        return a + b;
    } else if constexpr (std::is_same_v<T, half>) {
        return __hadd(a, b);                       // 半精度加法内置函数
    } else if constexpr (std::is_same_v<T, half2>) {
        return __hadd2(a, b);                      // 一条 SIMD 指令同时加两个 half
    } else if constexpr (std::is_same_v<T, half4>) {
        return make_half4(__hadd(a.x, b.x), __hadd(a.y, b.y),
                          __hadd(a.z, b.z), __hadd(a.w, b.w));
    }
}

// 示例：grid-stride 循环核函数（c[i] = add(a[i], b[i])）
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n, size_t step) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += step) {
        c[i] = add(a[i], b[i]);
    }
}

// 示例：host 包装函数
template <typename T>
void vector_add(T *c, const T *a, const T *b, size_t n, const dim3 &grid, const dim3 &block) {
    size_t step = grid.x * block.x;
    add_kernel<T><<<grid, block>>>(c, a, b, n, step);
}

int main() {
    // 步骤1：定义大小、初始化主机数据、分配设备显存
    const size_t SIZE = 1 << 20;                  // 1M 个 half 元素（每个 2 字节，数据量 2MB）
    size_t size_bytes = SIZE * sizeof(half);

    std::vector<half> h_a(SIZE, __float2half(1.0f));
    std::vector<half> h_b(SIZE, __float2half(2.0f));
    std::vector<half> h_c(SIZE, __float2half(0.0f));

    half *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, size_bytes));

    // 步骤2：将数据从主机拷贝到设备
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice));

    // 步骤3：配置核函数并调用
    dim3 block_dim(256);
    dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

    // 预热 2 次 + 计时 100 次，得到平均 kernel 耗时
    const int WARMUP_ITERS = 2;
    const int PROFILE_ITERS = 100;
    double avg_ms = cudabench::profile(WARMUP_ITERS, PROFILE_ITERS, [&] {
        vector_add(d_c, d_a, d_b, SIZE, grid_dim, block_dim);
    });
    CUDA_CHECK(cudaGetLastError());

    // 步骤4：将结果拷贝回主机
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));

    // 全量验证：每个元素都应为 3
    const float expected_value = 3.0f;
    bool verified = cudabench::verify_all(h_c, [](half v) {
        return __half2float(v) == 3.0f;
    });

    // 打印运行概要
    cudabench::Report report;
    report.vector_size   = SIZE;
    report.data_type     = "half";
    report.element_size  = sizeof(half);
    report.block_threads = block_dim.x;
    report.grid_blocks   = grid_dim.x;
    report.warmup_iters  = WARMUP_ITERS;
    report.profile_iters = PROFILE_ITERS;
    report.avg_ms        = avg_ms;
    report.verified      = verified;
    report.expected      = expected_value;
    report.got           = __half2float(h_c[0]);
    report.print();

    // 步骤5：释放显存
    if (d_a) CUDA_CHECK(cudaFree(d_a));
    if (d_b) CUDA_CHECK(cudaFree(d_b));
    if (d_c) CUDA_CHECK(cudaFree(d_c));

    return 0;
}

/* ================== 半精度要点（图1 / 图2） ==================
   - #include <cuda_fp16.h>：half / half2 类型及 __hadd / __hadd2 /
     __float2half / __half2float 等半精度支持都在这里。
   - half = 2 字节 fp16（1 符号位 + 5 指数位 + 10 尾数位），范围远小于 float；
   - half2 = 4 字节，把两个 half 打包，硬件有一条 SIMD 指令同时加两个；
   - __float2half()：float -> half；__half2float()：half -> float。

   课件（图2）写法：return {__hadd(a.x, b.x), __hadd(a.y, b.y)};
     —— __hadd 逐分量相加，花括号初始化构造 half2，写法正确；
   本文件等价写法：return __hadd2(a, b);
     —— 打包好的两个 half 用一条 SIMD 指令同时相加，指令更少；
   两种写法都合法、效果等价。
================================================================ */
