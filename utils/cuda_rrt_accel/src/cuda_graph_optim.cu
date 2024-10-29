#include "../include/cuda_basic.h"
#include "../include/class_def.h"
#include <math_constants.h>


__global__ static void __CUDA_KERNEL_optimize_graph_with_node(float4 *frame, int width, int height, int target_x, int target_z, int parent_x, int parent_z, float cost, float search_radius) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (frame[pos].w != 1.0) // w means that the point is part of the graph
        return;

    // printf("checking (%d, %d)\n",x, z);

    int dx = target_x - x;
    int dz = target_z - z;

    float dist_to_target = sqrtf(dx * dx + dz * dz);
    if (dist_to_target > search_radius) {
        // printf("[%d, %d] dist_to_target (%f) > search_radius (%f)\n", x, z, dist_to_target, search_radius);
        return;
    }

    int target_pos = target_z * width + target_x;
    
    float cost_with_target_as_parent = frame[target_pos].z + dist_to_target;

    if (cost_with_target_as_parent >= frame[pos].z) { // z is the stored cost for a graph node
        // printf("[%d, %d] cost_with_target_as_parent (%f) > curr cost (%f)\n", x, z, cost_with_target_as_parent, frame[pos].z);
        return;
    }

    // printf("success! optimizing (%d, %d), set parent to (%d, %d) with cost %f\n",
    //     x, z, target_x, target_z, cost_with_target_as_parent
    // );

    // we should change our parent to target
    frame[pos].x = target_x;
    frame[pos].y = target_z;
    frame[pos].z = cost_with_target_as_parent;
    frame[pos].w = 1.0;
}


void CUDA_optimize_graph_with_node(float4 *frame, int width, int height, int x, int z, int parent_x, int parent_z, float cost, float search_radius) {
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    __CUDA_KERNEL_optimize_graph_with_node<<<numBlocks, 256>>>(frame, width, height, x, z, parent_x, parent_z, cost, search_radius);
    CUDA(cudaDeviceSynchronize());
}