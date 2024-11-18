#include "cuda_basic.h"
#include "class_def.h"
#include <math_constants.h>

extern __global__ void __CUDA_KERNEL_optimizeGraphWithNode(double4 *graph, double * graph_cost, float3 *og, int *classCost, double *checkParams, double goal_heading, int parent_candidate_x, int parent_candidate_z, float radius);

void CUDA_optimizeGraphWithNode(double4 *graph, double * graph_cost, float3 *og, int *classCost, double *checkParams, double goal_heading, int x, int z, float radius)
{
    int width = static_cast<int>(checkParams[0]);
    int height = static_cast<int>(checkParams[1]);

    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    __CUDA_KERNEL_optimizeGraphWithNode<<<numBlocks, 256>>>(graph,  graph_cost, og, classCost, checkParams, goal_heading, x, z, radius);

    CUDA(cudaDeviceSynchronize());
}
