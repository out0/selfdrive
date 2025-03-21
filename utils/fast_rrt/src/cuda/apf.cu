#include "../../../cudac/include/cuda_basic.h"
#include "../../include/cuda_params.h"
#include "../../include/graph.h"
#include <math_constants.h>

#define FORCE_RANGE 5

extern __device__ __host__ void incIntrinsicCost(float3 *graphData, int width, int x, int z, double cost);

__global__ static void __CUDA_KERNEL_repulsive_force(float3 *og, float3 *graphData, float *classCosts, int *params, int width, int height, float Kr_half, int radius)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    float c = classCosts[TO_INT(og[pos].x)];
    if (c >= 0)
        return;  // not an obstacle

    for (int h = z - radius; h <= z + radius; h++)
    {
        if (h < 0)
            continue;
        
        if (h >= height)
            break;

        int init = x - radius;
        float p0 = (float)radius;

        for (int w = init; w <= x + radius; w++)
        {
            if (w < 0)
                continue;

            if (w >= width)
                break;

            if (w >= lower_bound_ego_x && w <= upper_bound_ego_x && h >= upper_bound_ego_z && h <= lower_bound_ego_z)
                continue;

            float p = sqrtf((w - x) * (w - x) + (h - z) * (h - z));
            if (p > p0)
                continue;

            float f = 1/p - 1/p0;
            float cost = Kr_half * f * f;
            incIntrinsicCost(graphData, width, w, h, cost);
            //printf("(%d, %d): r = %f, cost =  %f, 1/p = %f, 1/p0 = %f, f = %f\n", w, h, r, cost, 1/p, 1/p0, f);

            // printf("(%d, %d) cost = %f\n", w, h, cost);
        }
    }
}

void CudaGraph::computeAPF(float3 *og, float Kr, int radius)
{
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_repulsive_force<<<numBlocks, THREADS_IN_BLOCK>>>(
        og,
        _frameData->getCudaPtr(),
        _classCosts,
        _searchSpaceParams,
        _frame->width(),
        _frame->height(),
        Kr * 0.5,
        radius);

    CUDA(cudaDeviceSynchronize());
}
