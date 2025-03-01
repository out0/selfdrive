#include "../../include/cuda_grid.h"

__global__ static void __CUDA_KERNEL_Clear(double4 *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    frame[pos].x = 0.0;
    frame[pos].y = 0.0;
    frame[pos].z = 0.0;
    frame[pos].w = 0.0;
}
__global__ static void __CUDA_KERNEL_Clear(float4 *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    frame[pos].x = 0.0;
    frame[pos].y = 0.0;
    frame[pos].z = 0.0;
    frame[pos].w = 0.0;
}
__global__ static void __CUDA_KERNEL_Clear(int4 *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    frame[pos].x = 0.0;
    frame[pos].y = 0.0;
    frame[pos].z = 0.0;
    frame[pos].w = 0.0;
}

__global__ static void __CUDA_KERNEL_Clear(double3 *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    frame[pos].x = 0.0;
    frame[pos].y = 0.0;
    frame[pos].z = 0.0;
}
__global__ static void __CUDA_KERNEL_Clear(float3 *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    frame[pos].x = 0.0;
    frame[pos].y = 0.0;
    frame[pos].z = 0.0;
}
__global__ static void __CUDA_KERNEL_Clear(int3 *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    frame[pos].x = 0.0;
    frame[pos].y = 0.0;
    frame[pos].z = 0.0;
}

__global__ static void __CUDA_KERNEL_Clear(double2 *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    frame[pos].x = 0.0;
    frame[pos].y = 0.0;
}
__global__ static void __CUDA_KERNEL_Clear(float2 *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    frame[pos].x = 0.0;
    frame[pos].y = 0.0;
}
__global__ static void __CUDA_KERNEL_Clear(int2 *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    frame[pos].x = 0;
    frame[pos].y = 0;
}

__global__ static void __CUDA_KERNEL_Clear(double *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    frame[pos] = 0.0;
}
__global__ static void __CUDA_KERNEL_Clear(float *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    frame[pos] = 0.0;
}
__global__ static void __CUDA_KERNEL_Clear(int *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    frame[pos] = 0;
}
void CUDA_clear(double4 *frame, int width, int height)
{
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_KERNEL_Clear<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());
}
void CUDA_clear(float4 *frame, int width, int height)
{
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_KERNEL_Clear<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());
}
void CUDA_clear(int4 *frame, int width, int height)
{
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_KERNEL_Clear<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());
}
void CUDA_clear(double3 *frame, int width, int height)
{
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_KERNEL_Clear<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());
}
void CUDA_clear(float3 *frame, int width, int height)
{
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_KERNEL_Clear<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());
}
void CUDA_clear(int3 *frame, int width, int height)
{
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_KERNEL_Clear<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());
}
void CUDA_clear(double2 *frame, int width, int height)
{
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_KERNEL_Clear<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());
}
void CUDA_clear(float2 *frame, int width, int height)
{
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_KERNEL_Clear<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());
}
void CUDA_clear(int2 *frame, int width, int height)
{
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_KERNEL_Clear<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());
}

void CUDA_clear(double *frame, int width, int height)
{
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_KERNEL_Clear<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());
}
void CUDA_clear(float *frame, int width, int height)
{
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_KERNEL_Clear<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());
}
void CUDA_clear(int *frame, int width, int height)
{
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_KERNEL_Clear<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());
}

