#include <iostream>
#include <vector>
#include <cstdlib>
#include <type_traits>
#include <cuda_runtime.h>

// 步骤5（底部）：错误检查宏 + 运行概要工具（见 common/cuda_bench.cuh）
#include "../common/cuda_bench.cuh"

// 图2：__device__ 模板 add() —— 统一封装不同类型加法（仅在 GPU 上运行）
// float2：8 字节、8 字节对齐，make_float2 创建
template <typename T>
__device__ T add(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, float>) {
        return a + b;
    } else if constexpr (std::is_same_v<T, float2>) {
        return make_float2(a.x + b.x, a.y + b.y);
    } else if constexpr (std::is_same_v<T, float3>) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    } else if constexpr (std::is_same_v<T, float4>) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

// 图2：grid-stride 循环核函数（c[i] = add(a[i], b[i])）
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n, size_t step) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += step) {
        c[i] = add(a[i], b[i]);
    }
}

// 图2：host 包装函数（外层循环移到核函数内部）
template <typename T>
void vector_add(T *c, const T *a, const T *b, size_t n, const dim3 &grid, const dim3 &block) {
    size_t step = grid.x * block.x;
    add_kernel<T><<<grid, block>>>(c, a, b, n, step);
}

int main() {
    // 步骤1：定义大小、初始化主机数据、分配设备显存
    const size_t SIZE = 1 << 20;                  // 1M 个 float2 元素（每个 8 字节，数据量 8MB）
    size_t size_bytes = SIZE * sizeof(float2);

    std::vector<float2> h_a(SIZE, make_float2(1, 1));
    std::vector<float2> h_b(SIZE, make_float2(2, 2));
    std::vector<float2> h_c(SIZE, make_float2(0, 0));

    float2 *d_a, *d_b, *d_c;
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

    // 全量验证：每个元素的两个分量都应为 3
    const float expected_value = 3.0f;
    bool verified = cudabench::verify_all(h_c, [](float2 v) {
        return v.x == 3.0f && v.y == 3.0f;
    });

    // 打印运行概要
    cudabench::Report report;
    report.vector_size   = SIZE;
    report.data_type     = "float2";
    report.element_size  = sizeof(float2);
    report.block_threads = block_dim.x;
    report.grid_blocks   = grid_dim.x;
    report.warmup_iters  = WARMUP_ITERS;
    report.profile_iters = PROFILE_ITERS;
    report.avg_ms        = avg_ms;
    report.verified      = verified;
    report.expected      = expected_value;
    report.got           = h_c[0].x;
    report.print();

    // 步骤5：释放显存
    if (d_a) CUDA_CHECK(cudaFree(d_a));
    if (d_b) CUDA_CHECK(cudaFree(d_b));
    if (d_c) CUDA_CHECK(cudaFree(d_c));

    return 0;
}

/* ================== 循环展开参考：图3 显式特化写法 ==================
   把模板 add() 按类型“展开”成组件逐一访问的形式，效果等价。
   注意向量化类型内部子元素的访问方式：.x / .y / .z / .w

// Abstract kernel launcher function
template <typename T>
__global__ void vector_add_kernel(T *c, const T *a, const T *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Specialization for float2
template <>
__global__ void vector_add_kernel<float2>(float2 *c, const float2 *a, const float2 *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx].x = a[idx].x + b[idx].x;
        c[idx].y = a[idx].y + b[idx].y;
    }
}

// Specialization for float3（⚠ 反例：语法上能像 float2/float4 一样“展开”，
// 但内存合并与内存事务上有问题，详见文件末尾反例说明）
template <>
__global__ void vector_add_kernel<float3>(float3 *c, const float3 *a, const float3 *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx].x = a[idx].x + b[idx].x;
        c[idx].y = a[idx].y + b[idx].y;
        c[idx].z = a[idx].z + b[idx].z;
    }
}

// Specialization for float4
template <>
__global__ void vector_add_kernel<float4>(float4 *c, const float4 *a, const float4 *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx].x = a[idx].x + b[idx].x;
        c[idx].y = a[idx].y + b[idx].y;
        c[idx].z = a[idx].z + b[idx].z;
        c[idx].w = a[idx].w + b[idx].w;
    }
}

/* ---- float3 反例说明：内存合并 与 内存事务 ----
   ✗ 内存事务偏多：float3 = 12 字节、仅 4 字节对齐，CUDA 没有 12 字节的向量加载，
     编译器只能把 float3 拆成 8B + 4B（甚至 4B × 3）两次加载，LDG 指令与内存
     事务数比 float2/float4 多一倍。
   ✗ 内存合并度下降：一个 warp 读 32 个 float3 = 384 字节，这 384 字节横跨
     3~4 条 128B 缓存行且首尾不对齐、元素与 32B 扇区交错，按 32B 扇区服务内存
     事务时会多占/多读扇区，有效带宽利用率下降。
   ✓ 对照：float2 = 8B（warp 256B = 恰好 2 条 128B 行，对齐）
           float4 = 16B（warp 512B = 恰好 4 条 128B 行，对齐）
           二者都可单条 8B/16B 向量加载，事务数最少、合并度最高。
   结论：向量化一般只用 float2/float4；确实需要 12 字节时，用“补 4 字节填充
         成 float4（或 int4）”的办法，避免直接用 float3。
------------------------------------------------ */
================================================================= */
