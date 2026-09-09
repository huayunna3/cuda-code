#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

// 步骤5（底部）：错误检查宏 + 运行概要工具（见 common/cuda_bench.cuh）
#include "../common/cuda_bench.cuh"

// 步骤3（底部）：CUDA核函数定义（图1：单次只处理一块，偏移由外层循环传入）
template<typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n, size_t step) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + step; // step 实为外层循环传入的偏移
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 图1：主机端外层循环 + 偏移量调用核函数
template<typename T>
void vector_add(T *c, const T *a, const T *b, size_t n, const dim3 &grid, const dim3 &block) {
    size_t step = grid.x * block.x;
    for (size_t i = 0; i < n; i += step) {
        add_kernel<T><<<grid, block>>>(c, a, b, n, i);
    }
}

int main() {
    // 步骤1：定义大小、初始化主机数据、分配设备显存
    const size_t SIZE = 1 << 20; // ~1M elements
    size_t size_bytes = SIZE * sizeof(float);

    std::vector<float> h_a(SIZE, 1);
    std::vector<float> h_b(SIZE, 2);
    std::vector<float> h_c(SIZE, 0);

    float *d_a, *d_b, *d_c;
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

    // 图1：主机端外层循环 + 偏移量调用（step 逻辑已移入 vector_add 内部）
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
    bool verified = cudabench::verify_all(h_c, [](float v) { return v == 3.0f; });

    // 打印运行概要
    cudabench::Report report;
    report.vector_size   = SIZE;
    report.data_type     = "float";
    report.element_size  = sizeof(float);
    report.block_threads = block_dim.x;
    report.grid_blocks   = grid_dim.x;
    report.warmup_iters  = WARMUP_ITERS;
    report.profile_iters = PROFILE_ITERS;
    report.avg_ms        = avg_ms;
    report.verified      = verified;
    report.expected      = expected_value;
    report.got           = h_c[0];
    report.print();

    // 步骤5：释放显存
    if (d_a) CUDA_CHECK(cudaFree(d_a));
    if (d_b) CUDA_CHECK(cudaFree(d_b));
    if (d_c) CUDA_CHECK(cudaFree(d_c));

    return 0;
}
