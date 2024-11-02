#include "../include/cuda_basic.h"
#include "../include/class_def.h"
#include <math_constants.h>

#define NUM_POINTS_ON_MEAN 3

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

__device__ float __CUDA_KERNEL_compute_mean_heading(float4 *waypoints, int pos, int waypoints_count, bool *valid, int width, int height)
{
    float heading = 0.0;
    int count = 0;

    for (int j = 1; j <= NUM_POINTS_ON_MEAN; j++)
    {
        bool v = false;
        if (pos + j >= waypoints_count)
            break;
        heading += __CUDA_KERNEL_ComputeHeading((int)waypoints[pos].x, (int)waypoints[pos].y, (int)waypoints[pos + j].x, (int)waypoints[pos + j].y, &v, width, height);
        if (!v)
            break;
        count++;
    }

    if (count != NUM_POINTS_ON_MEAN)
    {
        count = 0;
        // compute in reverse
        for (int j = 1; j <= NUM_POINTS_ON_MEAN; j++)
        {
            bool v = false;
            if (pos - j < 0)
            {
                *valid = false;
                return 0.0;
            }
            heading += __CUDA_KERNEL_ComputeHeading((int)waypoints[pos - j].x, (int)waypoints[pos - j].y, (int)waypoints[pos].x, (int)waypoints[pos].y, &v, width, height);
            if (!v)
                break;
            count++;
        }
    }

    *valid = count > 0;

    if (*valid)
        return heading / count;

    return 0.0;
}

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

__global__ static void __CUDA_KERNEL_optimize_graph_with_node(
    float4 *graph,
    float3 *cuda_frame,
    int *classCost,
    int width,
    int height,
    int target_x,
    int target_z,
    int parent_x,
    int parent_z,
    float cost,
    float search_radius,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_ego_x,
    int lower_bound_ego_z,
    int upper_bound_ego_x,
    int upper_bound_ego_z)
{

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (x == target_x && z == target_z)
        return;

    if (graph[pos].w != 1.0) // w means that the point is part of the graph
        return;

    int dx = abs(target_x - x);
    int dz = abs(target_z - z);

    if (dz == 0) // pure left or right neighbors are not desired
        return;

    float dist_to_target = sqrtf(dx * dx + dz * dz);
    if (dist_to_target > search_radius)
        return;

    int target_pos = target_z * width + target_x;

    int target_parent_x = __float2int_rn(graph[target_pos].x);
    int target_parent_z = __float2int_rn(graph[target_pos].y);

    // I'm searching my parent, which is forbidden because it is a cyclic ref.
    if (target_parent_x == x && target_parent_z == z)
        return;

    float target_cost = graph[target_pos].z;
    float my_cost = graph[pos].z;

    if (my_cost < target_cost + dist_to_target)
        return;

    // lets check if I can connect to target

    int num_steps = __float2int_rn(max(dx, dz));
    if (num_steps == 0)
        return;

    float dxs = dx / num_steps;
    float dzs = dz / num_steps;

    int last_x = target_x;
    int last_z = target_z;

    bool *valid = new bool{false};

    if (valid == nullptr)
        return;

    float heading = __CUDA_KERNEL_ComputeHeading(parent_x, parent_z, target_x, target_z, valid, width, height);
    if (!valid)
        return;

    if (x < target_x)
    {
        dxs = -dxs;
    }
    if (z < target_z)
    {
        dzs = -dzs;
    }

    if (!__CUDA_KERNEL_ComputeFeasibleForAngle(
            cuda_frame, classCost,
            target_x,
            target_z,
            heading,
            width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        return;

    for (int i = 1; i < num_steps; i++)
    {
        int px = __float2int_rn(target_x + dxs * i);
        int pz = __float2int_rn(target_z + dzs * i);

        if (px == last_x && pz == last_z)
            continue;

        if (!__CUDA_KERNEL_ComputeFeasibleForAngle(
                cuda_frame, classCost,
                px,
                pz,
                heading,
                width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
            return;

        last_x = x;
        last_z = z;
    }

    // for (int i = 1; i < true_size; i++) {
    //     __CUDA_KERNEL_compute_mean_heading(connect_path, i, true_size, valid, width, height);
    // }
    // printf("optimized: (%d, %d) before (%d, %d) cost %f, after (%d, %d), cost %f\n",
    //        __float2int_rn(x), __float2int_rn(z), __float2int_rn(graph[pos].x),
    //        __float2int_rn(graph[pos].y), graph[pos].z, target_x, target_z,
    //        target_cost + dist_to_target);

    // we should change our parent to target
    graph[pos].x = target_x;
    graph[pos].y = target_z;
    graph[pos].z = target_cost + dist_to_target;
    graph[pos].w = 1.0;

    delete valid;
}

void CUDA_optimize_graph_with_node(
    float4 *graph,
    float3 *cuda_frame,
    int *classCost,
    int width, int height, int x, int z, int parent_x, int parent_z, float cost, float search_radius,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_ego_x,
    int lower_bound_ego_z,
    int upper_bound_ego_x,
    int upper_bound_ego_z)
{

    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    __CUDA_KERNEL_optimize_graph_with_node<<<numBlocks, 256>>>(graph, cuda_frame, classCost, width, height, x, z, parent_x, parent_z, cost, search_radius,
                                                               min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z);

    CUDA(cudaDeviceSynchronize());
}