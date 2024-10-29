#include "../include/cuda_basic.h"
#include "../include/class_def.h"
#include <math_constants.h>

__global__ static void __CUDA_KERNEL_Clear(float4 *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    frame[pos].x = 0.0;
    frame[pos].y = 0.0;
    frame[pos].z = 0.0;
    frame[pos].w = 0.0;
}

__global__ static void __CUDA_KERNEL_count_elements_in_graph(float4 *frame, int width, int height, unsigned int *count)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    if (frame[pos].w == 1.0)
    {
        // int z = pos / width;
        // int x = pos - z * width;
        // printf("%d, %d is in graph, inc count...\n", x, z);
        atomicInc(count, width * height);
    }
}

__global__ static void __CUDA_KERNEL_check_in_graph(float4 *frame, int width, int height, int target_x, int target_z, unsigned int *count)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (x != target_x || z != target_z)
        return;

    if (frame[pos].w == 1.0)
    {
        // printf("check in graph success for %d, %d\n", x, z);
        atomicInc(count, width * height);
    }
}

void CUDA_clear(float4 *frame, int width, int height)
{
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    __CUDA_KERNEL_Clear<<<numBlocks, 256>>>(frame, width, height);

    CUDA(cudaDeviceSynchronize());
}

/*
    BEST NEIGHBOR
*/

__global__ void __CUDA_KERNEL_find_best_neighbor_cost(float4 *frame, int width, int height, int target_x, int target_z, float radius, int *bestCost)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (frame[pos].w != 1.0) // w means that the point is part of the graph
        return;

    int dx = target_x - x;
    int dz = target_z - z;

    int dist = dx * dx + dz * dz;

    int r = radius * radius;

    if (dist > r)
        return;

    // self cost + dist
    int cost =__float2int_rn(sqrtf(dist) + frame[pos].z);

    atomicMin(bestCost, cost);
}

__global__ void __CUDA_KERNEL_find_waypoint_with_best_cost(float4 *frame, int width, int height, int target_x, int target_z, float radius, int *bestCost, int3 *point)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (frame[pos].w != 1.0) // z means that the point is part of the graph
        return;

    int dx = target_x - x;
    int dz = target_z - z;

    int dist = dx * dx + dz * dz;
    int cost =__float2int_rn(sqrtf(dist) + frame[pos].z);


    if (cost == *bestCost)
    {
        point->x = x;
        point->y = z;
        point->z = 1;
    }
}

int *CUDA_find_best_neighbor(float4 *frame, int3 *point, int width, int height, int goal_x, int goal_z, float radius)
{
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    int *bestCost;
    if (!cudaAllocMapped(&bestCost, sizeof(int)))
    {
        fprintf(stderr, "[CUDA RRT] unable to allocate %ld bytes for best cost in CUDA_find_best_neighbor()\n", sizeof(int));
        return nullptr;
    }

    *bestCost = 999999999;
    __CUDA_KERNEL_find_best_neighbor_cost<<<numBlocks, 256>>>(frame, width, height, goal_x, goal_z, radius, bestCost);
    CUDA(cudaDeviceSynchronize());

    if (*bestCost < 999999999)
    {
        __CUDA_KERNEL_find_waypoint_with_best_cost<<<numBlocks, 256>>>(frame, width, height, goal_x, goal_z, radius, bestCost, point);
        CUDA(cudaDeviceSynchronize());
    }

    int *res = nullptr;

    if (point->z >= 1.0)
        res = new int[3]{point->x, point->y, 1};
    else
        res = new int[3]{0, 0, 0};

    cudaFreeHost(bestCost);

    return res;
}

/*
    NEAREST NEIGHBOR
*/

__global__ void __CUDA_KERNEL_find_nearest_neighbor_dist(float4 *frame, int width, int height, int target_x, int target_z, int *bestDistance)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (frame[pos].w != 1.0) // w means that the point is part of the graph
        return;

    int dx = target_x - x;
    int dz = target_z - z;

    int dist = dx * dx + dz * dz;

    atomicMin(bestDistance, dist);
}

__global__ void __CUDA_KERNEL_find_waypoint_with_nearest_dist(float4 *frame, int width, int height, int target_x, int target_z, int *bestDistance, int3 *point)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (frame[pos].w != 1.0) // z means that the point is part of the graph
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

int *CUDA_find_nearest_neighbor(float4 *frame, int3 *point, int width, int height, int goal_x, int goal_z)
{
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    int *bestCost;
    if (!cudaAllocMapped(&bestCost, sizeof(int)))
    {
        fprintf(stderr, "[CUDA RRT] unable to allocate %ld bytes for best cost in CUDA_find_best_neighbor()\n", sizeof(int));
        return nullptr;
    }

    *bestCost = 999999999;
    __CUDA_KERNEL_find_nearest_neighbor_dist<<<numBlocks, 256>>>(frame, width, height, goal_x, goal_z, bestCost);
    CUDA(cudaDeviceSynchronize());

    if (*bestCost < 999999999)
    {
        __CUDA_KERNEL_find_waypoint_with_nearest_dist<<<numBlocks, 256>>>(frame, width, height, goal_x, goal_z, bestCost, point);
        CUDA(cudaDeviceSynchronize());
    }

    int *res = nullptr;

    if (point->z >= 1.0)
        res = new int[3]{point->x, point->y, 1};
    else
        res = new int[3]{0, 0, 0};

    cudaFreeHost(bestCost);

    return res;
}

unsigned int CUDA_count_elements_in_graph(float4 *frame, int width, int height)
{
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    unsigned int *count;
    if (!cudaAllocMapped(&count, sizeof(unsigned int)))
    {
        fprintf(stderr, "[CUDA RRT] unable to allocate %ld bytes for counting elements in graph\n", sizeof(unsigned int));
        return 0;
    }

    *count = 0;
    __CUDA_KERNEL_count_elements_in_graph<<<numBlocks, 256>>>(frame, width, height, count);
    CUDA(cudaDeviceSynchronize());

    unsigned int res = *count;

    // printf ("CUDA_count_elements_in_graph => %d\n", res);

    cudaFreeHost(count);
    return res;
}

bool CUDA_check_in_graph(float4 *frame, int width, int height, int x, int z)
{
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    unsigned int *count;
    if (!cudaAllocMapped(&count, sizeof(unsigned int)))
    {
        fprintf(stderr, "[CUDA RRT] unable to allocate %ld bytes for counting elements in graph\n", sizeof(unsigned int));
        return 0;
    }

    *count = 0;
    __CUDA_KERNEL_check_in_graph<<<numBlocks, 256>>>(frame, width, height, x, z, count);
    CUDA(cudaDeviceSynchronize());

    int res = *count;

    cudaFreeHost(count);
    return res > 0;
}


__global__ static void __CUDA_KERNEL_list_elements_in_graph(float4 *frame, float *res, int width, int height, unsigned int *list_pos)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (frame[pos].w == 1.0)
    {
        int store_pos = 5 * atomicInc(list_pos, width * height);
        res[store_pos] = x;
        res[store_pos+1] = z;
        res[store_pos+2] = frame[pos].x;
        res[store_pos+3] = frame[pos].y;
        res[store_pos+4] = frame[pos].z;
    }
}


void CUDA_list_elements(float4 *frame, 
    float *result,
    int width,
    int height,
    int count)
{

    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    float *cudaResult;
    if (!cudaAllocMapped(&cudaResult, sizeof(float) * count * 5))
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
        frame,
        cudaResult, 
        width, 
        height, 
        listPos);

    CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < *listPos * 5; i++) {
        result[i] = cudaResult[i];
    }

    cudaFreeHost(cudaResult);
    cudaFreeHost(listPos);

}