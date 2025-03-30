
#include "../../../cudac/include/cuda_basic.h"
#include "../../include/cuda_params.h"
#include "../../include/graph.h"
#include <math_constants.h>

__global__ static void __CUDA_KERNEL_count_elements_in_range(int3 *graph, int width, int height, int type, int xp, int zp, float radius_sqr, unsigned int *count)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (x == xp && z == zp) return;

    float dx = x - xp;
    float dz = z - zp;

    if (dx * dx + dz * dz > radius_sqr)
        return;

    if (graph[pos].z == type)
    {
        // int z = pos / width;
        // int x = pos - z * width;
        // printf("%d, %d is in graph, inc count...\n", x, z);
        atomicInc(count, width * height);
    }
}

unsigned int CudaGraph::__countInRange(int xp, int zp, float radius_sqr)
{
    int size = _frame->width() * _frame->height();

    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_parallelCount = 0;
    __CUDA_KERNEL_count_elements_in_range<<<numBlocks, THREADS_IN_BLOCK>>>(_frame->getCudaPtr(), _frame->width(), _frame->height(), GRAPH_TYPE_NODE, xp, zp, radius_sqr, _parallelCount);
    cudaDeviceSynchronize();

    return *_parallelCount;
}

__global__ static void __CUDA_KERNEL_list_elements_in_range(int3 *graph, int width, int height, int type, int xp, int zp, float radius_sqr, int2 *res, unsigned int *currentPos)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (x == xp && z == zp) return;

    float dx = x - xp;
    float dz = z - zp;
    if (dx * dx + dz * dz > radius_sqr)
        return;

    if (graph[pos].z == type)
    {
        int store_pos = atomicInc(currentPos, width * height);
        res[store_pos].x = x;
        res[store_pos].y = z;
    }
}

std::pair<int2 *, int> CudaGraph::__listNodesInRange(int type, int x, int z, float radius_sqr)
{
    unsigned int c = __countInRange(x, z, radius_sqr);

    int size = _frame->width() * _frame->height();

    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    int2 *cudaResult = nullptr;

    size_t nodeListSize = sizeof(int2) * (c + 1);

    if (!cudaAllocMapped(&cudaResult, nodeListSize))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(nodeListSize) + std::string(" bytes for listing\n");
        throw msg;
    }

    unsigned int *listPos = nullptr;
    if (!cudaAllocMapped(&listPos, sizeof(unsigned int)))
    {
        cudaFreeHost(cudaResult);
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(unsigned int)) + std::string(" bytes for listing\n");
        throw msg;
    }
    *listPos = 0;

    __CUDA_KERNEL_list_elements_in_range<<<numBlocks, THREADS_IN_BLOCK>>>(
        _frame->getCudaPtr(), 
        _frame->width(), 
        _frame->height(), 
        type,
        x,
        z,
        radius_sqr,
        cudaResult, 
        listPos);
    CUDA(cudaDeviceSynchronize());

    cudaFreeHost(listPos);

    return {cudaResult, c};
}

std::vector<int2> CudaGraph::listInRange(int x, int z, float radius)
{
    std::pair<int2 *, int> lst = __listNodesInRange(GRAPH_TYPE_NODE, x, z, radius * radius);
    int count = lst.second;
    int2 *cudaResult = lst.first;

    std::vector<int2> res;
    for (int i = 0; i < count; i++)
    {
        res.push_back({cudaResult[i].x, cudaResult[i].y});
    }

    cudaFreeHost(cudaResult);

    return res;
}