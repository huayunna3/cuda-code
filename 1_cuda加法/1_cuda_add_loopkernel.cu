#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

// 步骤5（底部）：错误检查宏定义
#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
    { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << "\n"; \
        exit(1); \
    } \
}

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
    vector_add(d_c, d_a, d_b, SIZE, grid_dim, block_dim);

    // 强烈建议加上这行以捕获核函数启动时的潜在错误
    CUDA_CHECK(cudaGetLastError());

    // 步骤4：将结果拷贝回主机
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));

    // 验证结果（原图没有，但可以加上以证明运行成功）
    std::cout << "计算结果验证 (a[0]+b[0]): " << h_c[0] << " (预期: 3)" << std::endl;
    std::cout << "计算结果验证 (a[last]+b[last]): " << h_c[SIZE-1] << " (预期: 3)" << std::endl;

    // 步骤5：释放显存
    if (d_a) CUDA_CHECK(cudaFree(d_a));
    if (d_b) CUDA_CHECK(cudaFree(d_b));
    if (d_c) CUDA_CHECK(cudaFree(d_c));

    return 0;
}
