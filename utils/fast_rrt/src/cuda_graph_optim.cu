#include "cuda_basic.h"
#include "class_def.h"
#include <math_constants.h>


/* ----------------- GLOBAL -------------------------------- */

__global__ void __CUDA_KERNEL_find_nearest_neighbor_dist(float4 *graph, float3 *frame, int width, int height, int target_x, int target_z, int *bestDistance)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].w != 1.0) // w means that the point is part of the graph
        return;

    int dx = target_x - x;
    int dz = target_z - z;

    int dist = __float2int_rn(sqrtf(dx * dx + dz * dz));

    atomicMin(bestDistance, dist);
}

/* ----------------- CALLING -------------------------------- */

// int2 CUDA_find_nearest_feasible_neighbor(float4 *frame, int width, int height)
// {
// }