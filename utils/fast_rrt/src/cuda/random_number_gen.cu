#include <curand_kernel.h>
#include <chrono>
#include "../../include/graph.h"

__global__ void __CUDA_KERNEL_setupRandomGenKernel(curandState *state, int size, long long seed){

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < size) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__device__ float generateRandom(curandState* state, int pos, float max) {
    return  max * curand_uniform(&state[pos]);
}

__device__ float generateRandomNeg(curandState* state, int pos, float max) {
    return  (2.0f * curand_uniform(&state[pos]) - 1.0f) * max;
}

    
long long getCurrentTimeMillis() {
    // Get current time_point
    auto now = std::chrono::system_clock::now();
    // Convert to duration since epoch
    auto duration = now.time_since_epoch();
    // Convert duration to milliseconds and return as long long
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

void CudaGraph::__initializeRandomGenerator()
{   
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    if (cudaMalloc(&this->_randState, sizeof(curandState) * size) != cudaSuccess)
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(curandState) * size) + std::string(" bytes for random number generation\n");
        throw msg;
    }

    __CUDA_KERNEL_setupRandomGenKernel<<<numBlocks, THREADS_IN_BLOCK>>>(
        _randState, size, getCurrentTimeMillis());

    CUDA(cudaDeviceSynchronize());
}