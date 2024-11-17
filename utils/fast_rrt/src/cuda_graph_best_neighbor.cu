#include "cuda_basic.h"
#include "class_def.h"

extern __global__ void __CUDA_KERNEL_find_feasible_lowest_neighbor_cost(double4 *graph, double *graph_cost, float3 *og, int *classCost, double *checkParams, int target_x, int target_z, float radius, long long *bestCost);
extern __global__ void __CUDA_KERNEL_find_feasible_neighbor_with_cost(double4 *graph, double *graph_cost, float3 *og, int *classCost, double *checkParams, int target_x, int target_z, long long *bestCost, int3 *point);

__global__ void __CUDA_KERNEL_find_lowest_neighbor_cost(double4 *graph, double *graph_cost, int width, int height, int target_x, int target_z, float radius, long long *bestCost)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].w != 1.0) // w means that the point is part of the graph
        return;

    int dx = target_x - x;
    int dz = target_z - z;

    long long dist = __float2ll_rd(dx * dx + dz * dz);

    long long r = __float2ll_rd(radius * radius);

    if (dist > r)
        return;

    // self cost + dist
    long long cost = __float2ll_rd(sqrtf(dist) + graph_cost[pos]);

    atomicMin(bestCost, cost);
}

__global__ void __CUDA_KERNEL_find_neighbor_with_cost(double4 *graph, double *graph_cost, int width, int height, int target_x, int target_z, float radius, long long *bestCost, int3 *point)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].w != 1.0) // w means that the point is part of the graph
        return;

    int dx = target_x - x;
    int dz = target_z - z;

    long long cost = __float2ll_rd(sqrtf(dx * dx + dz * dz) + graph_cost[pos]);

    if (cost == *bestCost)
    {
        point->x = x;
        point->y = z;
        point->z = 1;
    }
}

int2 CUDA_find_best_neighbor(double4 *graph, double *graph_cost, int3 *point, long long *bestValue, int width, int height, int x, int z, float radius)
{

    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    *bestValue = 999999999;
    __CUDA_KERNEL_find_lowest_neighbor_cost<<<numBlocks, 256>>>(graph, graph_cost, width, height, x, z, radius, bestValue);
    CUDA(cudaDeviceSynchronize());

    point->z = 0.0;

    if (*bestValue < 999999999)
    {
        __CUDA_KERNEL_find_neighbor_with_cost<<<numBlocks, 256>>>(graph, graph_cost, width, height, x, z, radius, bestValue, point);
        CUDA(cudaDeviceSynchronize());
    }

    int2 ret;
    ret.x = -1;
    ret.y = -1;

    if (point->z >= 1.0)
    {
        ret.x = point->x;
        ret.y = point->y;
    }

    return ret;
}

int2 CUDA_find_best_feasible_neighbor(double4 *graph, double *graph_cost, float3 *og, int *classCost, double *checkParams, int3 *point, long long *bestValue, int x, int z, float radius)
{
    int width = static_cast<int>(checkParams[0]);
    int height = static_cast<int>(checkParams[1]);
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    *bestValue = 999999999;
    __CUDA_KERNEL_find_feasible_lowest_neighbor_cost<<<numBlocks, 256>>>(graph, graph_cost, og, classCost, checkParams, x, z, radius, bestValue);
    CUDA(cudaDeviceSynchronize());

    point->z = 0.0;

    if (*bestValue < 999999999)
    {
        __CUDA_KERNEL_find_feasible_neighbor_with_cost<<<numBlocks, 256>>>(graph, graph_cost, og, classCost, checkParams, x, z, bestValue, point);
        CUDA(cudaDeviceSynchronize());
    }

    int2 ret;
    ret.x = -1;
    ret.y = -1;

    if (point->z >= 1.0)
    {
        ret.x = point->x;
        ret.y = point->y;
    }

    return ret;
}