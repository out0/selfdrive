#include "cuda_basic.h"
#include "class_def.h"

__global__ static void __CUDA_KERNEL_Clear(double4 *graph, double *graph_cost, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    graph[pos].x = 0.0;
    graph[pos].y = 0.0;
    graph[pos].z = 0.0;
    graph[pos].w = 0.0;
    graph_cost[pos] = 0.0;
}

void CUDA_clear(double4 *graph, double *graph_cost, int width, int height)
{
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    __CUDA_KERNEL_Clear<<<numBlocks, 256>>>(graph, graph_cost, width, height);

    CUDA(cudaDeviceSynchronize());
}
