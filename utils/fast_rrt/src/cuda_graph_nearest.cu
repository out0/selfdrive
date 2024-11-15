#include "cuda_basic.h"
#include "class_def.h"

extern __global__ void __CUDA_KERNEL_find_nearest_feasible_neighbor_dist(double4 *graph, float3 *og, int *classCost, double *checkParams, int target_x, int target_z, long long *bestDistance);


__global__ void __CUDA_KERNEL_find_nearest_neighbor_dist(double4 *graph, int width, int height, int target_x, int target_z, long long *bestDistance)
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

    long dist = dx * dx + dz * dz;

    atomicMin(bestDistance, dist);
}


__global__ void __CUDA_KERNEL_find_waypoint_with_nearest_dist(double4 *frame, int width, int height, int target_x, int target_z, long long *bestDistance, int3 *point)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (frame[pos].w != 1.0) // w means that the point is part of the graph
        return;

    int dx = target_x - x;
    int dz = target_z - z;

    int dist = dx * dx + dz * dz;

    if (dist == *bestDistance)
    {
        point->x = x;
        point->y = z;
        point->z = 1;
    }
}


int2 CUDA_find_nearest_neighbor(double4 *graph, int3* point, long long *bestValue, int width, int height, int x, int z) {

    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    *bestValue = 999999999;
    __CUDA_KERNEL_find_nearest_neighbor_dist<<<numBlocks, 256>>>(graph, width, height, x, z, bestValue);
    CUDA(cudaDeviceSynchronize());

    point->z = 0.0;

    if (*bestValue < 999999999)
    {
        __CUDA_KERNEL_find_waypoint_with_nearest_dist<<<numBlocks, 256>>>(graph, width, height, x, z, bestValue, point);
        CUDA(cudaDeviceSynchronize());
    }

    int2 ret;
    ret.x = -1;
    ret.y = -1;

    if (point->z >= 1.0) {
        ret.x = point->x;
        ret.y = point->y;
    }

    return ret;
}


int2 CUDA_find_nearest_feasible_neighbor(double4 *graph, float3 *og, int *classCost, double *checkParams, int3* point, long long *bestValue, int x, int z) {

    int width = static_cast<int>(checkParams[0]);
    int height = static_cast<int>(checkParams[1]);

    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    *bestValue = 999999999;
    __CUDA_KERNEL_find_nearest_feasible_neighbor_dist<<<numBlocks, 256>>>(graph, og, classCost, checkParams, x, z, bestValue);
    CUDA(cudaDeviceSynchronize());

    point->z = 0.0;

    if (*bestValue < 999999999)
    {
        __CUDA_KERNEL_find_waypoint_with_nearest_dist<<<numBlocks, 256>>>(graph, width, height, x, z, bestValue, point);
        CUDA(cudaDeviceSynchronize());
    }

    int2 ret;
    ret.x = -1;
    ret.y = -1;

    if (point->z >= 1.0) {
        ret.x = point->x;
        ret.y = point->y;
    }

    return ret;
}