#include "cuda_basic.h"
#include "class_def.h"
#include <math_constants.h>


__global__ static void __CUDA_KERNEL_count_elements_in_graph(double4 *graph, int width, int height, unsigned int *count)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    if (graph[pos].w == 1.0)
    {
        // int z = pos / width;
        // int x = pos - z * width;
        // printf("%d, %d is in graph, inc count...\n", x, z);
        atomicInc(count, width * height);
    }
}

unsigned int CUDA_parallel_count(double4 *graph, unsigned int *pcount, int width, int height)
{
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    *pcount = 0;
    __CUDA_KERNEL_count_elements_in_graph<<<numBlocks, 256>>>(graph, width, height, pcount);
    CUDA(cudaDeviceSynchronize());

    unsigned int res = *pcount;

    // printf ("CUDA_count_elements_in_graph => %d\n", res);

    return res;
}