#include "cuda_basic.h"
#include "class_def.h"
#include <math_constants.h>


__global__ static void __CUDA_KERNEL_count_elements_in_graph(double4 *graph, int width, int height, unsigned int *count)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
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



__global__ static void __CUDA_KERNEL_list_elements_in_graph(double4 *graph, double *graph_cost, double *res, int width, int height, unsigned int *list_pos)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].w == 1.0)
    {
        int store_pos = 6 * atomicInc(list_pos, width * height);
        printf ("storing list element starting on pos %d\n", store_pos);
        res[store_pos] = x;
        res[store_pos+1] = z;
        res[store_pos+2] = graph[pos].z; // heading
        res[store_pos+3] = graph[pos].x; // parent X
        res[store_pos+4] = graph[pos].y; // parent Z
        res[store_pos+5] = graph_cost[pos]; // cost
    }
}

void CUDA_list_elements(double4 *graph,
    double *graph_cost,
    double *result,
    int width,
    int height,
    int count)
{

    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    double *cudaResult;
    if (!cudaAllocMapped(&cudaResult, sizeof(double) * (count+1) * 6))
    {
        fprintf(stderr, "[CUDA Graph] unable to allocate %ld bytes for list elements in CUDA_list_elements()\n", sizeof(float) * count * 5);
        return;
    }

    unsigned int *listPos;
    if (!cudaAllocMapped(&listPos, sizeof(unsigned int)))
    {
        fprintf(stderr, "[CUDA Graph] unable to allocate %ld bytes for list elements position in CUDA_list_elements()\n", sizeof(unsigned int));
        cudaFreeHost(cudaResult);
        return;
    }
    *listPos = 0;

    __CUDA_KERNEL_list_elements_in_graph<<<numBlocks, 256>>>(
        graph,
        graph_cost,
        cudaResult, 
        width, 
        height, 
        listPos);

    CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < count * 6; i++) {
        result[i] = cudaResult[i];
    }

    cudaFreeHost(cudaResult);
    cudaFreeHost(listPos);

}