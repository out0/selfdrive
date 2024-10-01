#include "../include/cuda_basic.h"
#include "../include/class_def.h"
#include <math_constants.h>

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
                // if (x == DEBUG_X && z == DEBUG_Z)
                // {
                //     printf("(%d, %d) not feasible on angle %f because of position: (%d, %d)\n", x, z, angle_radians * 180 / CUDART_PI_F, xl, zl);
                //     printf("(%d, %d) min distances: W: %d  H: %d\n",  x, z, min_dist_x, min_dist_z);
                // }
                return false;
            }
        }

    // if (x == DEBUG_X && z == DEBUG_Z)
    // {
    //     printf("(%d, %d) feasible on angle %f\n", x, z, angle_radians * CUDART_PI_F / 180);
    // }
    return true;
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

__device__ static float __CUDA_KERNEL_ComputeHeading_Unbound_Values(int p1_x, int p1_y, int p2_x, int p2_y, bool *valid, int width, int height)
{
    *valid = false;
    if (p1_x == p2_x && p1_y == p2_y)
        return 0.0;

    float dx = p2_x - p1_x;
    float dz = p2_y - p1_y;
    *valid = true;
    float heading = CUDART_PI_F / 2 - atan2f(-dz, dx);

    if (heading > CUDART_PI_F) // greater than 180 deg
        heading = heading - 2 * CUDART_PI_F;

    return heading;
}

#ifdef DEBUG_SET_GOAL_VECTOR
__device__ static void __CUDA_KERNEL_FrameColor_Debug(float3 *frame, int width, int height, uchar3 *classColors)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int y = pos / width;
    int x = pos - y * width;

    if (y >= height)
        return;
    if (x >= width)
        return;

    int segClass = frame[pos].x;

    frame[pos].x = classColors[segClass].x;
    frame[pos].y = classColors[segClass].y;
    frame[pos].z = classColors[segClass].z;
}
#endif

__global__ static void __CUDA_KERNEL_bestWaypointCostForHeading(float3 *frame, int *params, int *classCost, float angle, int *bestCost)
{
    int width = params[0];
    int height = params[1];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    // if (x == DEBUG_X && z == DEBUG_Z)
    // {
    //     printf("(%d, %d) investigating set of headings\n", x, z);
    // }

    int goal_x = params[2];
    int goal_z = params[3];

#ifdef MINIMAL_DISTANCE_X
    int min_dist_x = MINIMAL_DISTANCE_X;
#else
    int min_dist_x = params[4];
#endif

#ifdef MINIMAL_DISTANCE_Z
    int min_dist_z = MINIMAL_DISTANCE_Z;
#else
    int min_dist_z = params[5];
#endif

    int lower_bound_ego_x = params[6];
    int lower_bound_ego_z = params[7];
    int upper_bound_ego_x = params[8];
    int upper_bound_ego_z = params[9];

    int dx = goal_x - x;
    int dz = goal_z - z;

    int cost = (dx * dx + dz * dz) % 10000000;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (classCost[(int)frame[pos].x] < 0)
        return;

    if (!__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, angle, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        return;

    atomicMin(bestCost, cost);
}

__global__ static void __CUDA_KERNEL_bestWaypointPos(float3 *frame, int *params, int *classCost, int *bestCost)
{
    int width = params[0];
    int height = params[1];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    // if (x == DEBUG_X && z == DEBUG_Z)
    // {
    //     printf("(%d, %d) investigating set of headings\n", x, z);
    // }

    int goal_x = params[2];
    int goal_z = params[3];

#ifdef MINIMAL_DISTANCE_X
    int min_dist_x = MINIMAL_DISTANCE_X;
#else
    int min_dist_x = params[4];
#endif

#ifdef MINIMAL_DISTANCE_Z
    int min_dist_z = MINIMAL_DISTANCE_Z;
#else
    int min_dist_z = params[5];
#endif

    int lower_bound_ego_x = params[6];
    int lower_bound_ego_z = params[7];
    int upper_bound_ego_x = params[8];
    int upper_bound_ego_z = params[9];

    int dx = goal_x - x;
    int dz = goal_z - z;

    int cost = (dx * dx + dz * dz) % 10000000;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (classCost[(int)frame[pos].x] < 0)
        return;

    bool valid = false;

    float angle = __CUDA_KERNEL_ComputeHeading_Unbound_Values(x, z, goal_x, goal_z, &valid, width, height);
    // float angle_d = 180 * angle / CUDART_PI_F;

    if (!valid)
    {
        // printf ("invalid\n");
        return;
    }

    if (!__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, angle, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        return;

    // if (cost < 410)
    //     printf("[bestWaypointPos] (%d, %d) feasible angle = %f (%f degrees) with cost %d\n", x, z, angle, angle_d, cost);

    atomicMin(bestCost, cost);
}

__global__ static void __CUDA_KERNEL_findWaypointForCostAndHeading(float3 *frame, int *params, int *classCost, int *bestCost, float angle, int *waypoint)
{
    int width = params[0];
    int height = params[1];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    // if (x == DEBUG_X && z == DEBUG_Z)
    // {
    //     printf("(%d, %d) investigating set of headings\n", x, z);
    // }

    int goal_x = params[2];
    int goal_z = params[3];

#ifdef MINIMAL_DISTANCE_X
    int min_dist_x = MINIMAL_DISTANCE_X;
#else
    int min_dist_x = params[4];
#endif

#ifdef MINIMAL_DISTANCE_Z
    int min_dist_z = MINIMAL_DISTANCE_Z;
#else
    int min_dist_z = params[5];
#endif

    int lower_bound_ego_x = params[6];
    int lower_bound_ego_z = params[7];
    int upper_bound_ego_x = params[8];
    int upper_bound_ego_z = params[9];

    int dx = goal_x - x;
    int dz = goal_z - z;

    int cost = (dx * dx + dz * dz) % 10000000;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (cost == *bestCost)
    {
        if (!__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, angle, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
            return;

        waypoint[0] = x;
        waypoint[1] = z;
    }

    // atomicCAS(posForCost, cost == *bestCost, pos);
}

__global__ static void __CUDA_KERNEL_findBestHeadingForCost(float3 *frame, int *params, int *classCost, int *bestCost, int *bestHeading)
{
    int width = params[0];
    int height = params[1];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    // if (x == DEBUG_X && z == DEBUG_Z)
    // {
    //     printf("(%d, %d) investigating set of headings\n", x, z);
    // }

    int goal_x = params[2];
    int goal_z = params[3];

#ifdef MINIMAL_DISTANCE_X
    int min_dist_x = MINIMAL_DISTANCE_X;
#else
    int min_dist_x = params[4];
#endif

#ifdef MINIMAL_DISTANCE_Z
    int min_dist_z = MINIMAL_DISTANCE_Z;
#else
    int min_dist_z = params[5];
#endif

    int lower_bound_ego_x = params[6];
    int lower_bound_ego_z = params[7];
    int upper_bound_ego_x = params[8];
    int upper_bound_ego_z = params[9];

    int dx = goal_x - x;
    int dz = goal_z - z;

    int cost = (dx * dx + dz * dz) % 10000000;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (cost == *bestCost)
    {
        bool valid = false;
        float angle = __CUDA_KERNEL_ComputeHeading_Unbound_Values(x, z, goal_x, goal_z, &valid, width, height);

        if (!valid)
            return;

        if (!__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, angle, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
            return;

        // printf("[CUDA] best heading found with cost %d = %f\n", cost, angle);
        int new_value = (int)(10000 * angle);
        // int old_value = atomicMin(bestHeading, new_value);
        atomicMin(bestHeading, new_value);
        // printf("[CUDA] replacing old value %d = by new value %d\n", old_value, *bestHeading);
    }

    // atomicCAS(posForCost, cost == *bestCost, pos);
}

int *CUDA_bestWaypointPosForHeading(float3 *frame, int width, int height, int goal_x, int goal_z, float angle, int min_dist_x, int min_dist_z, int lower_bound_x, int lower_bound_z, int upper_bound_x, int upper_bound_z)
{
    const int numClasses = 29;
    int size = width * height;

    int *params = nullptr;
    if (!cudaAllocMapped(&params, sizeof(int) * 10))
    {
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for params at CUDA_bestWaypointPosForHeading()\n", sizeof(int) * 10);
        return nullptr;
    }

    int *costs = nullptr;
    if (!cudaAllocMapped(&costs, sizeof(int) * numClasses))
    {
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for class costs at CUDA_bestWaypointPosForHeading()\n", sizeof(int) * numClasses);
        cudaFreeHost(params);
        return nullptr;
    }

    params[0] = width;
    params[1] = height;
    params[2] = goal_x;
    params[3] = goal_z;
    params[4] = min_dist_x / 2;
    params[5] = min_dist_z / 2;
    params[6] = lower_bound_x;
    params[7] = lower_bound_z;
    params[8] = upper_bound_x;
    params[9] = upper_bound_z;

    // printf("bounds %d, %d -> %d, %d\n", lower_bound_x, lower_bound_z, upper_bound_x, upper_bound_z);

    for (int i = 0; i < numClasses; i++)
        costs[i] = segmentationClassCost[i];

    int numBlocks = floor(size / 256) + 1;

    int *bestCost;
    if (!cudaAllocMapped(&bestCost, sizeof(int)))
    {
        cudaFreeHost(params);
        cudaFreeHost(costs);
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for bestCost at CUDA_bestWaypointPosForHeading()\n", sizeof(int));
        return nullptr;
    }
    *bestCost = 999999;

    int *waypoint;
    if (!cudaAllocMapped(&waypoint, sizeof(int) * 2))
    {
        cudaFreeHost(params);
        cudaFreeHost(costs);
        cudaFreeHost(bestCost);
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for waypoint at CUDA_bestWaypointPosForHeading()\n", sizeof(int) * 2);
        return nullptr;
    }

    __CUDA_KERNEL_bestWaypointCostForHeading<<<numBlocks, 256>>>(frame, params, costs, angle, bestCost);
    CUDA(cudaDeviceSynchronize());

    __CUDA_KERNEL_findWaypointForCostAndHeading<<<numBlocks, 256>>>(frame, params, costs, bestCost, angle, waypoint);
    CUDA(cudaDeviceSynchronize());

    cudaFreeHost(params);
    cudaFreeHost(costs);
    cudaFreeHost(bestCost);
    cudaFreeHost(waypoint);

    return new int[2]{
        waypoint[0], waypoint[1]};
}

int *CUDA_bestWaypointPos(float3 *frame, int width, int height, int goal_x, int goal_z, int min_dist_x, int min_dist_z, int lower_bound_x, int lower_bound_z, int upper_bound_x, int upper_bound_z)
{
    const int numClasses = 29;
    int size = width * height;

    int *params = nullptr;
    if (!cudaAllocMapped(&params, sizeof(int) * 10))
    {
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for params at CUDA_bestWaypointPos()\n", sizeof(int) * numClasses);
        return nullptr;
    }

    int *costs = nullptr;
    if (!cudaAllocMapped(&costs, sizeof(int) * numClasses))
    {
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for class costs at CUDA_bestWaypointPos()\n", sizeof(int) * numClasses);
        cudaFreeHost(params);
        return nullptr;
    }

    params[0] = width;
    params[1] = height;
    params[2] = goal_x;
    params[3] = goal_z;
    params[4] = min_dist_x / 2;
    params[5] = min_dist_z / 2;
    params[6] = lower_bound_x;
    params[7] = lower_bound_z;
    params[8] = upper_bound_x;
    params[9] = upper_bound_z;

    // printf("bounds %d, %d -> %d, %d\n", lower_bound_x, lower_bound_z, upper_bound_x, upper_bound_z);

    for (int i = 0; i < numClasses; i++)
        costs[i] = segmentationClassCost[i];

    int numBlocks = floor(size / 256) + 1;

    int *bestCost;
    if (!cudaAllocMapped(&bestCost, sizeof(int)))
    {
        cudaFreeHost(params);
        cudaFreeHost(costs);
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for bestCost at CUDA_bestWaypointPos()\n", sizeof(int));
        return nullptr;
    }

    *bestCost = 999999;
    __CUDA_KERNEL_bestWaypointPos<<<numBlocks, 256>>>(frame, params, costs, bestCost);
    CUDA(cudaDeviceSynchronize());
    //    printf("chosen cost: %d\n", *bestCost);

    int *bestHeading;
    if (!cudaAllocMapped(&bestHeading, sizeof(int)))
    {
        cudaFreeHost(params);
        cudaFreeHost(costs);
        cudaFreeHost(bestCost);
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for bestHeading at CUDA_bestWaypointPos()\n", sizeof(int) * 2);
        return nullptr;
    }

    int *waypoint;
    if (!cudaAllocMapped(&waypoint, sizeof(int) * 2))
    {
        cudaFreeHost(params);
        cudaFreeHost(costs);
        cudaFreeHost(bestCost);
        cudaFreeHost(bestHeading);
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for waypoint at CUDA_bestWaypointPos()\n", sizeof(int) * 2);
        return nullptr;
    }

    *bestHeading = 999999;
    __CUDA_KERNEL_findBestHeadingForCost<<<numBlocks, 256>>>(frame, params, costs, bestCost, bestHeading);
    CUDA(cudaDeviceSynchronize());

    int headingInt = *bestHeading;
    float heading = (float)headingInt / 10000;
    // printf("bestHeading = %d, chosen heading: %f\n", *bestHeading, heading);

    __CUDA_KERNEL_findWaypointForCostAndHeading<<<numBlocks, 256>>>(frame, params, costs, bestCost, heading, waypoint);
    CUDA(cudaDeviceSynchronize());

    cudaFreeHost(params);
    cudaFreeHost(costs);
    cudaFreeHost(bestCost);
    cudaFreeHost(bestHeading);
    cudaFreeHost(waypoint);

    return new int[3]{
        waypoint[0], waypoint[1], 0};
}

__device__ int __CUDA_KERNEL_computeDirection(int start_x, int start_z, int goal_x, int goal_z)
{
    int direction = 0;

    // PURE LEFT / RIGHT DOESNT EXIST IN NON-HOLONOMIC SYSTEMS
    if (goal_z <= start_z)
        direction = direction | TOP;
    else if (goal_z > start_z)
        direction = direction | BOTTOM;

    if (goal_x < start_x)
        direction = direction | LEFT;
    else if (goal_x > start_x)
        direction = direction | RIGHT;

    return direction;
}

__device__ static float __CUDA_ToRadians(float angle)
{
    return (angle * CUDART_PI_F) / 180;
}

__global__ static void __CUDA_KERNEL_WaypointInDirection_findBestHeadingError(float3 *frame, int *params, int *classCost, int *bestHeadingError)
{
    int width = params[0];
    int height = params[1];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    // if (x == DEBUG_X && z == DEBUG_Z)
    // {
    //     printf("(%d, %d) investigating set of headings\n", x, z);
    // }

    int goal_x = params[2];
    int goal_z = params[3];

#ifdef MINIMAL_DISTANCE_X
    int min_dist_x = MINIMAL_DISTANCE_X;
#else
    int min_dist_x = params[4];
#endif

#ifdef MINIMAL_DISTANCE_Z
    int min_dist_z = MINIMAL_DISTANCE_Z;
#else
    int min_dist_z = params[5];
#endif

    int lower_bound_ego_x = params[6];
    int lower_bound_ego_z = params[7];
    int upper_bound_ego_x = params[8];
    int upper_bound_ego_z = params[9];
    int start_x = params[10];
    int start_z = params[11];

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (classCost[(int)frame[pos].x] < 0)
        return;

    bool valid = false;
    float best_heading = __CUDA_KERNEL_ComputeHeading_Unbound_Values(x, z, goal_x, goal_z, &valid, width, height);
    // float angle_d = 180 * angle / CUDART_PI_F;

    if (!valid)
        return;

    int direction = __CUDA_KERNEL_computeDirection(start_x, start_z, x, z);

    if (direction & BOTTOM)
        return;

    float best_local_cost = -1;

    if (direction & LEFT)
    {
        float angle = -90;
        while (angle <= 0)
        {
            float a = __CUDA_ToRadians(angle);
            if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, angle, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
            {
                float err = abs(best_heading - a);
                best_local_cost = best_local_cost < err ? best_local_cost : err;
            }
            angle += 5;
        }
    }
    if (direction & RIGHT)
    {
        float angle = 0;
        while (angle <= 90)
        {
            float a = __CUDA_ToRadians(angle);
            if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, a, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
            {
                float err = abs(best_heading - a);
                best_local_cost = best_local_cost < err ? best_local_cost : err;
            }
            angle += 5;
        }
    }

    if (best_local_cost < 0)
        return;

    atomicMin(bestHeadingError, 10000 * best_local_cost);
}

__global__ static void __CUDA_KERNEL_WaypointInDirection_findBestCostError(float3 *frame, int *params, int *classCost, int *bestHeadingError, int *bestCostError)
{
    int width = params[0];
    int height = params[1];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    // if (x == DEBUG_X && z == DEBUG_Z)
    // {
    //     printf("(%d, %d) investigating set of headings\n", x, z);
    // }

    int goal_x = params[2];
    int goal_z = params[3];

#ifdef MINIMAL_DISTANCE_X
    int min_dist_x = MINIMAL_DISTANCE_X;
#else
    int min_dist_x = params[4];
#endif

#ifdef MINIMAL_DISTANCE_Z
    int min_dist_z = MINIMAL_DISTANCE_Z;
#else
    int min_dist_z = params[5];
#endif

    int lower_bound_ego_x = params[6];
    int lower_bound_ego_z = params[7];
    int upper_bound_ego_x = params[8];
    int upper_bound_ego_z = params[9];
    int start_x = params[10];
    int start_z = params[11];

    int dx = goal_x - x;
    int dz = goal_z - z;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (classCost[(int)frame[pos].x] < 0)
        return;

    bool valid = false;

    float best_heading = __CUDA_KERNEL_ComputeHeading_Unbound_Values(x, z, goal_x, goal_z, &valid, width, height);
    // float angle_d = 180 * angle / CUDART_PI_F;

    if (!valid)
        return;

    int direction = __CUDA_KERNEL_computeDirection(start_x, start_z, goal_x, goal_z);

    if (direction & BOTTOM)
        return;

    if (direction & LEFT)
    {
        float angle = -90;
        while (angle <= 0)
        {
            float a = __CUDA_ToRadians(angle);
            int err = 10000 * abs(best_heading - a);
            if (err > *bestHeadingError)
                continue;

            if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, a, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
            {
                int cost = (dx * dx + dz * dz) % 10000000;
                atomicMin(bestCostError, cost);
                return;
            }
            angle += 5;
        }
    }
    if (direction & RIGHT)
    {
        float angle = 0;
        while (angle <= 90)
        {
            float a = __CUDA_ToRadians(angle);
            int err = 10000 * abs(best_heading - a);
            if (err > *bestHeadingError)
                continue;

            if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, a, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
            {
                int cost = (dx * dx + dz * dz) % 10000000;
                atomicMin(bestCostError, cost);
                return;
            }
            angle += 5;
        }
    }
}

__global__ static void __CUDA_KERNEL_WaypointInDirection_findWaypoint(float3 *frame, int *params, int *classCost, int *bestCost, int *bestHeading, int *waypoint)
{
    int width = params[0];
    int height = params[1];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    // if (x == DEBUG_X && z == DEBUG_Z)
    // {
    //     printf("(%d, %d) investigating set of headings\n", x, z);
    // }

    int goal_x = params[2];
    int goal_z = params[3];

#ifdef MINIMAL_DISTANCE_X
    int min_dist_x = MINIMAL_DISTANCE_X;
#else
    int min_dist_x = params[4];
#endif

#ifdef MINIMAL_DISTANCE_Z
    int min_dist_z = MINIMAL_DISTANCE_Z;
#else
    int min_dist_z = params[5];
#endif

    int lower_bound_ego_x = params[6];
    int lower_bound_ego_z = params[7];
    int upper_bound_ego_x = params[8];
    int upper_bound_ego_z = params[9];
    int start_x = params[10];
    int start_z = params[11];

    int dx = goal_x - x;
    int dz = goal_z - z;

    int cost = (dx * dx + dz * dz) % 10000000;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (cost == *bestCost)
    {
        bool valid = false;
        float best_heading = __CUDA_KERNEL_ComputeHeading_Unbound_Values(x, z, goal_x, goal_z, &valid, width, height);

        if (!valid)
            return;

        int direction = __CUDA_KERNEL_computeDirection(start_x, start_z, goal_x, goal_z);

        if (direction & BOTTOM)
            return;

        if (direction & LEFT)
        {
            float angle = -90;
            while (angle <= 0)
            {
                float a = __CUDA_ToRadians(angle);
                int err = 10000 * abs(best_heading - a);
                if (err > *bestHeading)
                    continue;

                if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, a, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
                {
                    waypoint[0] = x;
                    waypoint[1] = z;
                    waypoint[2] = angle;
                    return;
                }
                angle += 5;
            }
        }
        if (direction & RIGHT)
        {
            float angle = 0;
            while (angle <= 90)
            {
                float a = __CUDA_ToRadians(angle);
                int err = 10000 * abs(best_heading - a);
                if (err > *bestHeading)
                    continue;

                if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, a, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
                {
                    waypoint[0] = x;
                    waypoint[1] = z;
                    waypoint[2] = angle;
                    return;
                }
                angle += 5;
            }
        }
    }
}

int *CUDA_bestWaypointInDirection(float3 *frame, int width, int height, int start_x, int start_z, int goal_x, int goal_z,
                                  int min_dist_x, int min_dist_z, int lower_bound_x, int lower_bound_z, int upper_bound_x, int upper_bound_z)
{
    const int numClasses = 29;
    int size = width * height;

    int *params = nullptr;
    if (!cudaAllocMapped(&params, sizeof(int) * 12))
    {
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for params at CUDA_bestWaypointInDirection()\n", sizeof(int) * numClasses);
        return nullptr;
    }

    int *costs = nullptr;
    if (!cudaAllocMapped(&costs, sizeof(int) * numClasses))
    {
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for class costs at CUDA_bestWaypointInDirection()\n", sizeof(int) * numClasses);
        cudaFreeHost(params);
        return nullptr;
    }

    params[0] = width;
    params[1] = height;
    params[2] = goal_x;
    params[3] = goal_z;
    params[4] = min_dist_x / 2;
    params[5] = min_dist_z / 2;
    params[6] = lower_bound_x;
    params[7] = lower_bound_z;
    params[8] = upper_bound_x;
    params[9] = upper_bound_z;
    params[10] = start_x;
    params[11] = start_z;

    // printf("bounds %d, %d -> %d, %d\n", lower_bound_x, lower_bound_z, upper_bound_x, upper_bound_z);

    for (int i = 0; i < numClasses; i++)
        costs[i] = segmentationClassCost[i];

    int numBlocks = floor(size / 256) + 1;

    // STEP1: Find the best heading error, which narrows the pixels to those who can reach the goal and that have the lowest heading error possible:
    // the error is computed as err = abs (best_angle - a)
    // where best_angle = direct angle between x,z and the goal
    //       a = angle being analyzed for feasibility in direction   (if x,z is TOP-LEFT, a is in (-90, 0). If x,z is TOP-RIGHT, a is in (0, 90))
    int *bestHeadingError;
    if (!cudaAllocMapped(&bestHeadingError, sizeof(float)))
    {
        cudaFreeHost(params);
        cudaFreeHost(costs);
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for bestCost at CUDA_bestWaypointInDirection()\n", sizeof(int));
        return nullptr;
    }
    *bestHeadingError = 999999;
    __CUDA_KERNEL_WaypointInDirection_findBestHeadingError<<<numBlocks, 256>>>(frame, params, costs, bestHeadingError);
    CUDA(cudaDeviceSynchronize());

    printf("best heading error found for (%d, %d) -> (%d, %d): %d\n", start_x, start_z, goal_x, goal_z, *bestHeadingError);

    // STEP2: For all the points that are feasible at the bestHeadingError, find what is the closest distance between the point and the goal.

    int *bestCostError;
    if (!cudaAllocMapped(&bestCostError, sizeof(int)))
    {
        cudaFreeHost(params);
        cudaFreeHost(costs);
        cudaFreeHost(bestHeadingError);
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for bestCostError at CUDA_bestWaypointInDirection()\n", sizeof(int) * 2);
        return nullptr;
    }

    *bestCostError = 999999;
    __CUDA_KERNEL_WaypointInDirection_findBestCostError<<<numBlocks, 256>>>(frame, params, costs, bestHeadingError, bestCostError);
    CUDA(cudaDeviceSynchronize());

    printf("best cost error for the heading error is: %d\n", *bestCostError);

    int *waypoint;
    if (!cudaAllocMapped(&waypoint, sizeof(int) * 2))
    {
        cudaFreeHost(params);
        cudaFreeHost(costs);
        cudaFreeHost(bestCostError);
        cudaFreeHost(bestHeadingError);
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for waypoint at CUDA_bestWaypointInDirection()\n", sizeof(int) * 2);
        return nullptr;
    }
    // STEP3: Now we find the first goal point that has the lowest bestCostError and bestHeadingError
    __CUDA_KERNEL_WaypointInDirection_findWaypoint<<<numBlocks, 256>>>(frame, params, costs, bestCostError, bestHeadingError, waypoint);
    CUDA(cudaDeviceSynchronize());

    printf("that resulted in the first waypoing being: %d, %d, %d\n", waypoint[0], waypoint[1], waypoint[2]);

    int *res = new int[3]{
        waypoint[0], waypoint[1], waypoint[2]};

    cudaFreeHost(waypoint);
    return res;
}

#define NUM_POINTS_ON_MEAN 3

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

__global__ static void __CUDA_KERNEL_checkFeasibleWaypoints(float3 *frame, int *params, int *classCost, float4 *waypoints, int count, bool computeHeadings)
{
    int width = params[0];
    int height = params[1];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > count)
        return;

    int x = waypoints[pos].x;
    int z = waypoints[pos].y;
    // float heading = waypoints[pos].z;

    waypoints[pos].w = 1;

#ifdef MINIMAL_DISTANCE_X
    int min_dist_x = MINIMAL_DISTANCE_X;
#else
    int min_dist_x = params[2];
#endif

#ifdef MINIMAL_DISTANCE_Z
    int min_dist_z = MINIMAL_DISTANCE_Z;
#else
    int min_dist_z = params[3];
#endif

    int lower_bound_ego_x = params[4];
    int lower_bound_ego_z = params[5];
    int upper_bound_ego_x = params[6];
    int upper_bound_ego_z = params[7];

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    float heading = 0.0;
    if (computeHeadings)
    {
        bool valid = false;

        float heading = __CUDA_KERNEL_compute_mean_heading(waypoints, pos, count, &valid, width, height);

        waypoints[pos].z = heading;

        if (!valid)
            return;
    }
    else
    {
        heading = waypoints[pos].z;
    }

    // if (x == 116 && z == 3)
    // {
    //      printf("\n[GPU] heading for %d, %d = %f, pos = %d\n\n", x, z, heading, pos);
    //  }

    // if (x == DEBUG_X && z == DEBUG_Z)
    // {
    //     printf("(%d, %d) heading = %f rad %f deg  computed to point (%d, %d)\n", x, z, heading, 180 * heading / CUDART_PI_F, (int)waypoints[pos+1].x, (int)waypoints[pos+1].y);
    // }

    if (!__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, heading, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        waypoints[pos].w = 0;
}

float4 *CUDA_checkFeasibleWaypoints(
    float *points,
    int count,
    float3 *frame,
    int width,
    int height,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_x,
    int lower_bound_z,
    int upper_bound_x,
    int upper_bound_z,
    bool computeHeadings)
{
    const int numClasses = 29;
    int size = width * height;

    int *params = nullptr;
    if (!cudaAllocMapped(&params, sizeof(int) * 10))
    {
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for params at checkFeasibleWaypoints()\n", sizeof(int) * numClasses);
        return nullptr;
    }

    int *costs = nullptr;
    if (!cudaAllocMapped(&costs, sizeof(int) * numClasses))
    {
        cudaFreeHost(params);
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for class costs at checkFeasibleWaypoints()\n", sizeof(int) * numClasses);
        return nullptr;
    }

    float4 *waypoints;
    if (!cudaAllocMapped(&waypoints, sizeof(float4) * count))
    {
        cudaFreeHost(params);
        cudaFreeHost(costs);
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for waypoints at checkFeasibleWaypoints()\n", sizeof(float4) * count);
        return nullptr;
    }
    for (int c = 0; c < count; c++)
    {
        int pos = 4 * c;
        waypoints[c].x = points[pos];
        waypoints[c].y = points[pos + 1];
        waypoints[c].z = points[pos + 2];
        waypoints[c].w = 0;
    }

    params[0] = width;
    params[1] = height;
    params[2] = min_dist_x / 2;
    params[3] = min_dist_z / 2;
    params[4] = lower_bound_x;
    params[5] = lower_bound_z;
    params[6] = upper_bound_x;
    params[7] = upper_bound_z;

    // printf("bounds %d, %d -> %d, %d\n", lower_bound_x, lower_bound_z, upper_bound_x, upper_bound_z);

    for (int i = 0; i < numClasses; i++)
        costs[i] = segmentationClassCost[i];

    int numBlocks = floor(size / 256) + 1;

    __CUDA_KERNEL_checkFeasibleWaypoints<<<numBlocks, 256>>>(frame, params, costs, waypoints, count, computeHeadings);

    CUDA(cudaDeviceSynchronize());
    cudaFreeHost(params);
    cudaFreeHost(costs);
    return waypoints;
    // cudaFreeHost(waypoints);
}

__global__ static void __CUDA_KERNEL_ComputeCost(float3 *frame, int *params, int *classCost)
{
    int width = params[0];
    int height = params[1];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int goal_x = params[2];
    int goal_z = params[3];
    int min_dist_x = params[4];
    int min_dist_z = params[5];
    int lower_bound_ego_x = params[6];
    int lower_bound_ego_z = params[7];
    int upper_bound_ego_x = params[8];
    int upper_bound_ego_z = params[9];

    int dx = goal_x - x;
    int dz = goal_z - z;

    frame[pos].y = (float)sqrtf(dx * dx + dz * dz);
    frame[pos].z = 1;

    // Self-ego area should always be feasible and cost nothing.
    // z is inverted: upper is towards (-), lower is towards (+)
    //   (0,0) -----------> x
    //    |
    //    |
    //    |
    //    |
    //    \/ z
    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
    {
        frame[pos].z = 1;
        return;
    }

    int lower_range_z = max(0, z - min_dist_z);
    int upper_range_z = min(height - 1, z + min_dist_z);
    int lower_range_x = max(0, x - min_dist_x);
    int upper_range_x = min(width - 1, x + min_dist_x);

    for (int i = lower_range_z; i <= upper_range_z; i++)
        for (int j = lower_range_x; j <= upper_range_x; j++)
        {
            if (j >= lower_bound_ego_x && j <= upper_bound_ego_x && i >= upper_bound_ego_z && i <= lower_bound_ego_z)
                continue;

            int p = i * width + j;
            int c = frame[p].x;

            if (classCost[c] < 0)
            {
                frame[pos].z = 0;
                return;
            }
        }
}

void CUDA_setGoal(float3 *frame, int width, int height, int goal_x, int goal_z, int min_dist_x, int min_dist_z, int lower_bound_x, int lower_bound_z, int upper_bound_x, int upper_bound_z)
{
    const int numClasses = 29;
    int size = width * height;

    int *params = nullptr;
    if (!cudaAllocMapped(&params, sizeof(int) * 10))
        return;

    int *costs = nullptr;
    if (!cudaAllocMapped(&costs, sizeof(int) * numClasses))
    {
        cudaFreeHost(params);
        return;
    }

    params[0] = width;
    params[1] = height;
    params[2] = goal_x;
    params[3] = goal_z;
    params[4] = min_dist_x / 2;
    params[5] = min_dist_z / 2;
    params[6] = lower_bound_x;
    params[7] = lower_bound_z;
    params[8] = upper_bound_x;
    params[9] = upper_bound_z;

    // printf("bounds %d, %d -> %d, %d\n", lower_bound_x, lower_bound_z, upper_bound_x, upper_bound_z);

    for (int i = 0; i < numClasses; i++)
        costs[i] = segmentationClassCost[i];

    int numBlocks = floor(size / 256) + 1;

    __CUDA_KERNEL_ComputeCost<<<numBlocks, 256>>>(frame, params, costs);

    CUDA(cudaDeviceSynchronize());
    cudaFreeHost(params);
    cudaFreeHost(costs);
}

__global__ static void __CUDA_KERNEL_SetGoalVectorized(float3 *frame, int *params, int *classCost, uchar3 *classColorsDebug)
{
    int width = params[0];
    int height = params[1];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    // if (x == DEBUG_X && z == DEBUG_Z)
    // {
    //     printf("(%d, %d) investigating set of headings\n", x, z);
    // }

    int goal_x = params[2];
    int goal_z = params[3];

#ifdef MINIMAL_DISTANCE_X
    int min_dist_x = MINIMAL_DISTANCE_X;
#else
    int min_dist_x = params[4];
#endif

#ifdef MINIMAL_DISTANCE_Z
    int min_dist_z = MINIMAL_DISTANCE_Z;
#else
    int min_dist_z = params[5];
#endif

    int lower_bound_ego_x = params[6];
    int lower_bound_ego_z = params[7];
    int upper_bound_ego_x = params[8];
    int upper_bound_ego_z = params[9];

    int dx = goal_x - x;
    int dz = goal_z - z;

    frame[pos].y = (float)sqrtf(dx * dx + dz * dz);
    frame[pos].z = 0x0F;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
    {
        return;
    }

    if (classCost[(int)frame[pos].x] < 0)
    {
        // if (x == DEBUG_X && z == DEBUG_Z)
        // {
        //     printf("(%d, %d) classcost not feasible\n", x, z);
        // }
        frame[pos].z = 0x00;
        return;
    }

    int v = 0;

#ifdef DEBUG_SET_GOAL_VECTOR
    __CUDA_KERNEL_FrameColor_Debug(frame, width, height, classColorsDebug);
#endif

    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, ANGLE_HEADING_0, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_0;

    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, ANGLE_HEADING_22_5, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_22_5;

    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, ANGLE_HEADING_45, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_45;

    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, ANGLE_HEADING_67_5, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_67_5;

    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, ANGLE_HEADING_90, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_90;

    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, ANGLE_HEADING_MINUS_22_5, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_MINUS_22_5;

    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, ANGLE_HEADING_MINUS_45, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_MINUS_45;

    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, ANGLE_HEADING_MINUS_67_5, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_MINUS_67_5;

    frame[pos].z = v;
}

void CUDA_setGoalVectorized(float3 *frame, int width, int height, int goal_x, int goal_z, int min_dist_x, int min_dist_z, int lower_bound_x, int lower_bound_z, int upper_bound_x, int upper_bound_z)
{
    const int numClasses = 29;
    int size = width * height;

    int *params = nullptr;
    if (!cudaAllocMapped(&params, sizeof(int) * 10))
    {
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for params at checkFeasibleWaypoints()\n", sizeof(int) * numClasses);
        return;
    }

    int *costs = nullptr;
    if (!cudaAllocMapped(&costs, sizeof(int) * numClasses))
    {
        fprintf(stderr, "[CUDA FRAME] unable to allocate %ld bytes for class costs at checkFeasibleWaypoints()\n", sizeof(int) * numClasses);
        cudaFreeHost(params);
        return;
    }

    params[0] = width;
    params[1] = height;
    params[2] = goal_x;
    params[3] = goal_z;
    params[4] = min_dist_x / 2;
    params[5] = min_dist_z / 2;
    params[6] = lower_bound_x;
    params[7] = lower_bound_z;
    params[8] = upper_bound_x;
    params[9] = upper_bound_z;

    // printf("bounds %d, %d -> %d, %d\n", lower_bound_x, lower_bound_z, upper_bound_x, upper_bound_z);

    for (int i = 0; i < numClasses; i++)
        costs[i] = segmentationClassCost[i];

    int numBlocks = floor(size / 256) + 1;

    uchar3 *classColors = nullptr;

#ifdef DEBUG_SET_GOAL_VECTOR
    if (!cudaAllocMapped(&classColors, sizeof(uchar3) * numClasses))
    {
        return;
    }

    for (int i = 0; i < numClasses; i++)
    {
        classColors[i].x = segmentationClassColors[i][0];
        classColors[i].y = segmentationClassColors[i][1];
        classColors[i].z = segmentationClassColors[i][2];
    }
#endif

    __CUDA_KERNEL_SetGoalVectorized<<<numBlocks, 256>>>(frame, params, costs, classColors);

    CUDA(cudaDeviceSynchronize());
    cudaFreeHost(params);
    cudaFreeHost(costs);
}
