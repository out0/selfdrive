#include <cuda_runtime.h>
#include <driveless/cuda_basic.h>
#include <string>
#define THREADS_IN_BLOCK 256

__global__ static void __kernel_check_ptr_values_int2(int2 *graph, bool *fail, int size)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= size * size)
        return;

    if (graph[pos].x != 1 || graph[pos].y != 2)
    {
        *fail = true;
    }
}

__global__ static void __kernel_check_ptr_values_float2(float2 *graph, bool *fail, int size)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= size * size)
        return;

    if (graph[pos].x != 1.0 || graph[pos].y != 2.0)
    {
        *fail = true;
    }
}

__global__ static void __kernel_check_ptr_values_double2(double2 *graph, bool *fail, int size)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= size * size)
        return;

    if (graph[pos].x != 1.0 || graph[pos].y != 2.0)
    {
        *fail = true;
    }
}

__global__ static void __kernel_check_ptr_values_int3(int3 *graph, bool *fail, int size)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= size * size)
        return;

    if (graph[pos].x != 1 || graph[pos].y != 2 || graph[pos].z != 3)
    {
        *fail = true;
    }
}

__global__ static void __kernel_check_ptr_values_float3(float3 *graph, bool *fail, int size)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= size * size)
        return;

    if (graph[pos].x != 1.0 || graph[pos].y != 2.0 || graph[pos].z != 3.0)
    {
        *fail = true;
    }
}

__global__ static void __kernel_check_ptr_values_double3(double3 *graph, bool *fail, int size)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= size * size)
        return;

    if (graph[pos].x != 1.0 || graph[pos].y != 2.0 || graph[pos].z != 3.0)
    {
        *fail = true;
    }
}

__global__ static void __kernel_check_ptr_values_int4(int4 *graph, bool *fail, int size)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= size * size)
        return;

    if (graph[pos].x != 1 || graph[pos].y != 2 || graph[pos].z != 3 || graph[pos].w != 4)
    {
        *fail = true;
    }
}

__global__ static void __kernel_check_ptr_values_float4(float4 *graph, bool *fail, int size)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= size * size)
        return;

    if (graph[pos].x != 1.0 || graph[pos].y != 2.0 || graph[pos].z != 3.0 || graph[pos].w != 4.0)
    {
        *fail = true;
    }
}

__global__ static void __kernel_check_ptr_values_double4(double4 *graph, bool *fail, int size)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= size * size)
        return;

    if (graph[pos].x != 1.0 || graph[pos].y != 2.0 || graph[pos].z != 3.0 || graph[pos].w != 4.0)
    {
        *fail = true;
    }
}



bool check_ptr_values_int2 (int2 *ptr, int size) {
    int s = size * size;
    int numBlocks = floor(s / THREADS_IN_BLOCK) + 1;

    bool *fail = nullptr;
    
    if (!cudaAllocMapped(&fail, sizeof(bool)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(bool))+ "\n";
        throw msg;
    }

    *fail = false;
    __kernel_check_ptr_values_int2<<<numBlocks, THREADS_IN_BLOCK>>>(ptr, fail, size);
    CUDA(cudaDeviceSynchronize());
    return !*fail;
}

bool check_ptr_values_float2 (float2 *ptr, int size) {
    int s = size * size;
    int numBlocks = floor(s / THREADS_IN_BLOCK) + 1;

    bool *fail = nullptr;
    
    if (!cudaAllocMapped(&fail, sizeof(bool)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(bool))+ "\n";
        throw msg;
    }

    *fail = false;
    __kernel_check_ptr_values_float2<<<numBlocks, THREADS_IN_BLOCK>>>(ptr, fail, size);
    CUDA(cudaDeviceSynchronize());
    return !*fail;
}

bool check_ptr_values_double2 (double2 *ptr, int size) {
    int s = size * size;
    int numBlocks = floor(s / THREADS_IN_BLOCK) + 1;

    bool *fail = nullptr;
    
    if (!cudaAllocMapped(&fail, sizeof(bool)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(bool))+ "\n";
        throw msg;
    }

    *fail = false;
    __kernel_check_ptr_values_double2<<<numBlocks, THREADS_IN_BLOCK>>>(ptr, fail, size);
    CUDA(cudaDeviceSynchronize());
    return !*fail;
}


bool check_ptr_values_int3 (int3 *ptr, int size) {
    int s = size * size;
    int numBlocks = floor(s / THREADS_IN_BLOCK) + 1;

    bool *fail = nullptr;
    
    if (!cudaAllocMapped(&fail, sizeof(bool)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(bool))+ "\n";
        throw msg;
    }

    *fail = false;
    __kernel_check_ptr_values_int3<<<numBlocks, THREADS_IN_BLOCK>>>(ptr, fail, size);
    CUDA(cudaDeviceSynchronize());
    return !*fail;
}

bool check_ptr_values_float3 (float3 *ptr, int size) {
    int s = size * size;
    int numBlocks = floor(s / THREADS_IN_BLOCK) + 1;

    bool *fail = nullptr;
    
    if (!cudaAllocMapped(&fail, sizeof(bool)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(bool))+ "\n";
        throw msg;
    }

    *fail = false;
    __kernel_check_ptr_values_float3<<<numBlocks, THREADS_IN_BLOCK>>>(ptr, fail, size);
    CUDA(cudaDeviceSynchronize());
    return !*fail;
}

bool check_ptr_values_double3 (double3 *ptr, int size) {
    int s = size * size;
    int numBlocks = floor(s / THREADS_IN_BLOCK) + 1;

    bool *fail = nullptr;
    
    if (!cudaAllocMapped(&fail, sizeof(bool)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(bool))+ "\n";
        throw msg;
    }

    *fail = false;
    __kernel_check_ptr_values_double3<<<numBlocks, THREADS_IN_BLOCK>>>(ptr, fail, size);
    CUDA(cudaDeviceSynchronize());
    return !*fail;
}


bool check_ptr_values_int4 (int4 *ptr, int size) {
    int s = size * size;
    int numBlocks = floor(s / THREADS_IN_BLOCK) + 1;

    bool *fail = nullptr;
    
    if (!cudaAllocMapped(&fail, sizeof(bool)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(bool))+ "\n";
        throw msg;
    }

    *fail = false;
    __kernel_check_ptr_values_int4<<<numBlocks, THREADS_IN_BLOCK>>>(ptr, fail, size);
    CUDA(cudaDeviceSynchronize());
    return !*fail;
}

bool check_ptr_values_float4 (float4 *ptr, int size) {
    int s = size * size;
    int numBlocks = floor(s / THREADS_IN_BLOCK) + 1;

    bool *fail = nullptr;
    
    if (!cudaAllocMapped(&fail, sizeof(bool)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(bool))+ "\n";
        throw msg;
    }

    *fail = false;
    __kernel_check_ptr_values_float4<<<numBlocks, THREADS_IN_BLOCK>>>(ptr, fail, size);
    CUDA(cudaDeviceSynchronize());
    return !*fail;
}

bool check_ptr_values_double4 (double4 *ptr, int size) {
    int s = size * size;
    int numBlocks = floor(s / THREADS_IN_BLOCK) + 1;

    bool *fail = nullptr;
    
    if (!cudaAllocMapped(&fail, sizeof(bool)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(bool))+ "\n";
        throw msg;
    }

    *fail = false;
    __kernel_check_ptr_values_double4<<<numBlocks, THREADS_IN_BLOCK>>>(ptr, fail, size);
    CUDA(cudaDeviceSynchronize());
    return !*fail;
}



