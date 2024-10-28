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

/*
LINK
*/

__device__ static bool __CUDA_KERNEL_ComputeFeasibleForAngle(
    float3 *frame,
    int *classCost,
    int x,
    int z,
    float angle_radians,
    int width,
    int height,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_ego_x,
    int lower_bound_ego_z,
    int upper_bound_ego_x,
    int upper_bound_ego_z)
{
    float c = cosf(angle_radians);
    float s = sinf(angle_radians);

    for (int i = -min_dist_z; i <= min_dist_z; i++)
        for (int j = -min_dist_x; j <= min_dist_x; j++)
        {
            int xl = __float2int_rn(j * c - i * s + x);
            int zl = __float2int_rn(j * s + i * c + z);

            if (xl < 0 || xl >= width)
                continue;

            if (zl < 0 || zl >= height)
                continue;

            if (xl >= lower_bound_ego_x && xl <= upper_bound_ego_x && zl >= upper_bound_ego_z && zl <= lower_bound_ego_z)
                continue;

            int segmentation_class = __float2int_rn(frame[zl * width + xl].x);

            if (classCost[segmentation_class] < 0)
            {
                // if (x == 115 && z == 16)
                // {
                //     printf("(%d, %d) not feasible on angle %f because of position: (%d, %d)\n", x, z, angle_radians * 180 / CUDART_PI_F, xl, zl);
                //     printf("(%d, %d) min distances: W: %d  H: %d\n",  x, z, min_dist_x, min_dist_z);
                // }
                return false;
            }
        }

    // if (x == 115 && z == 16)
    // {
    //     printf("(%d, %d) feasible on angle %f\n", x, z, angle_radians * 180 / CUDART_PI_F);
    // }
    return true;
}

static float __CPU_ComputeHeading(int p1_x, int p1_y, int p2_x, int p2_y, bool *valid, int width, int height)
{
    *valid = false;
    if (p1_x == p2_x && p1_y == p2_y)
        return 0.0;

    if (p1_x < 0 || p1_y < 0 || p2_x < 0 || p2_y < 0)
        return 0.0;

    if (p1_x >= width || p1_y >= height || p2_x >= width || p2_y >= height)
        return 0.0;

    float dx = p2_x - p1_x;
    float dz = p2_y - p1_y;
    *valid = true;
    float heading = CUDART_PI_F / 2 - atan2f(-dz, dx);

    if (heading > CUDART_PI_F) // greater than 180 deg
        heading = heading - 2 * CUDART_PI_F;

    return heading;
}

__device__ static float __CUDA_KERNEL_ComputeHeading(int p1_x, int p1_y, int p2_x, int p2_y, bool *valid, int width, int height)
{
    *valid = false;
    if (p1_x == p2_x && p1_y == p2_y)
        return 0.0;

    if (p1_x < 0 || p1_y < 0 || p2_x < 0 || p2_y < 0)
        return 0.0;

    if (p1_x >= width || p1_y >= height || p2_x >= width || p2_y >= height)
        return 0.0;

    float dx = p2_x - p1_x;
    float dz = p2_y - p1_y;
    *valid = true;
    float heading = CUDART_PI_F / 2 - atan2f(-dz, dx);

    if (heading > CUDART_PI_F) // greater than 180 deg
        heading = heading - 2 * CUDART_PI_F;

    return heading;
}

__global__ void __CUDA_KERNEL_link_self_to_graph(
    float4 *frame, 
    float3 *cuda_frame, 
    int width, 
    int height, 
    int *classCost,
    int min_dist_x, 
    int min_dist_z,
    int lower_bound_ego_x,
    int lower_bound_ego_z,
    int upper_bound_ego_x,
    int upper_bound_ego_z,
    int parent_x, 
    int parent_z, 
    float max_distance, 
    float heading)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (frame[pos].w == 1.0) // w means that the point is already part of the graph
        return;

    int dx = x - parent_x;
    int dz = z - parent_z;

    float dist_to_parent = sqrtf(dx * dx + dz * dz);

    //printf("LINK: (%d, %d) dist_to_parent = %f\n", x, z, dist_to_parent);

    if (dist_to_parent > max_distance)
        return;

    // x, z are in the radius candidate for being linked to parent

    bool valid = false;

    float curr_heading = __CUDA_KERNEL_ComputeHeading(parent_x, parent_z, x, z, &valid, width, height);


    if (!valid)
        return;

    //printf("LINK: (%d, %d) curr_heading = %f\n", x, z, curr_heading);


    if (curr_heading != heading)
        return;


    // if (!__CUDA_KERNEL_ComputeFeasibleForAngle(cuda_frame,
    //     classCost,
    //     x, z, curr_heading, width, height, 
    //     min_dist_x, 
    //     min_dist_z,
    //     lower_bound_ego_x,
    //     lower_bound_ego_z,
    //     upper_bound_ego_x,
    //     upper_bound_ego_z)) return;


    //printf("LINK: (%d, %d) feasible!\n", x, z);

    int pos_parent = parent_z * width + parent_x;
    printf("LINK: (%d, %d) pos_parent = (%d, %d) = %d\n", x, z, parent_x, parent_z, pos_parent);

    frame[pos].z = frame[pos_parent].z + dist_to_parent;
    frame[pos].w = 1.0;

    printf("LINK: (%d, %d) total cost = %f\n", x, z, frame[pos].z);
}

void CUDA_link(float4 *frame, 
    float3 *cuda_frame,
    int width,
    int height,
    int *classCosts,
    int min_dist_x, 
    int min_dist_z,
    int lower_bound_ego_x,
    int lower_bound_ego_z,
    int upper_bound_ego_x,
    int upper_bound_ego_z,    
    int parent_x,
    int parent_z,
    int x,
    int z)
{
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    bool valid = false;
    float heading = __CPU_ComputeHeading(parent_x, parent_z, x, z, &valid, width, height);
    if (!valid)
        return;

    int dx = (x - parent_x);
    int dz = (z - parent_z);
    int num_steps = max(dx, dz);

    float dxs = dx / num_steps;
    float dzs = dz / num_steps;

    int last_x = parent_x;
    int last_z = parent_z;

    int2 *path;
    if (!cudaAllocMapped(&path, sizeof(int2) * num_steps))
    {
        fprintf(stderr, "[CUDA Graph] unable to allocate %ld bytes for best cost in CUDA_find_best_neighbor()\n", sizeof(int2) * num_steps);
        return;
    }

    
    for (int i = 1; i < num_steps; i++) {
        int x = __float2int_rn(parent_x + i * dxs);
        int z = __float2int_rn(parent_z + i * dzs);

        if (x == last_x && z == last_z) continue;

        path[i - 1].x = x;
        path[i - 1].y = z;
    }

    float max_dist = sqrtf(dx * dx + dz * dz);

    __CUDA_KERNEL_link_self_to_graph<<<numBlocks, 256>>>(
        frame,
        cuda_frame, 
        width, 
        height, 
        classCosts, 
        min_dist_x, 
        min_dist_z,
        lower_bound_ego_x,
        lower_bound_ego_z,
        upper_bound_ego_x,
        upper_bound_ego_z,    
        parent_x, parent_z, max_dist, heading);

    CUDA(cudaDeviceSynchronize());
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
        int store_pos = atomicInc(list_pos, width * height);
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