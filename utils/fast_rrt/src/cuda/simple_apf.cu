#include "../../../cudac/include/cuda_basic.h"
#include "../../include/cuda_params.h"
#include "../../include/graph.h"
#include <math_constants.h>

#define FORCE_INTENSITY 0.25
#define FORCE_RANGE 5

extern __device__ __host__ void incIntrinsicCost(double3 *graphData, int width, int x, int z, double cost);

__global__ static void __CUDA_KERNEL_APF(float3 *og, double3 *graphData, float * classCosts, int width, int height, int radius)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;
    

    float c = classCosts[TO_INT(og[pos].x)];
    if (c >= 0)  return;

    float sqRadius = radius * radius;

   
    for (int h = z - radius; h <= z + radius; h++) {
        if (h < 0) continue;
        if (h >= height) break;

        for (int w = x - radius; x <= x + radius; x++) {
            if (w < 0) continue;
            if (w >= width) break;
    
            float r = (w - x) * (w - x) + (h - z) * (h - z);
            if (r <= sqRadius) {
                float cost = (sqRadius - r) * FORCE_INTENSITY;
                incIntrinsicCost(graphData, width, w, h, cost);
                printf("(%d, %d) cost = %f\n", w, h, cost);
            }            
        }
    }    
}

void CudaGraph::computeAPF(float3 *og, int radius)
{
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_APF<<<numBlocks, THREADS_IN_BLOCK>>>(
        og,
        _frameData->getCudaPtr(),
        _classCosts,
        _frame->width(),
        _frame->height(),
        radius);

    CUDA(cudaDeviceSynchronize());
}

