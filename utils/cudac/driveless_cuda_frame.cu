#include "cuda_basic.h"
#include "class_def.h"
#include <math_constants.h>

#define DEBUG_X 131
#define DEBUG_Z 78

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

__global__ static void __CUDA_KERNEL_FrameColor(float3 *frame, uchar3 *output, int width, int height, uchar3 *classColors)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int y = pos / width;
    int x = pos - y * width;

    if (y >= height)
        return;
    if (x >= width)
        return;

    int segClass = frame[pos].x;

    output[pos].x = classColors[segClass].x;
    output[pos].y = classColors[segClass].y;
    output[pos].z = classColors[segClass].z;
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

__device__ static float __CUDA_KERNEL_ComputeHeading(float4 &p1, float4 &p2, bool *valid, int width, int height)
{
    *valid = false;
    if (p1.x == p2.x && p1.y == p2.y)
        return 0.0;

    if (p1.x < 0 || p1.y < 0 || p2.x < 0 || p2.y < 0)
        return 0.0;

    if (p1.x >= width || p1.y >= height || p2.x >= width || p2.y >= height)
        return 0.0;

    float dx = p2.x - p1.x;
    float dz = p2.y - p1.y;
    *valid = true;
    float heading = CUDART_PI_F / 2 - atan2f(-dz, dx);

    if (heading > CUDART_PI_F) // greater than 180 deg
        heading = heading - 2 * CUDART_PI_F;

    return heading;
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
        heading += __CUDA_KERNEL_ComputeHeading(waypoints[pos], waypoints[pos + j], &v, width, height);
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
            if (pos - j < 0)  {
                *valid = false;
                return 0.0;
            }
            heading += __CUDA_KERNEL_ComputeHeading(waypoints[pos - j], waypoints[pos], &v, width, height);
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

__global__ static void __CUDA_KERNEL_checkFeasibleWaypoints(float3 *frame, int *params, int *classCost, float4 *waypoints, int count)
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

    bool valid = false;
    float heading = __CUDA_KERNEL_compute_mean_heading(waypoints, pos, count, &valid, width, height);

    // if (x == 116 && z == 3)
    // {
    //      printf("\n[GPU] heading for %d, %d = %f, pos = %d\n\n", x, z, heading, pos);
    //  }

    waypoints[pos].z = heading;

    if (!valid)
        return;

    // if (x == DEBUG_X && z == DEBUG_Z)
    // {
    //     printf("(%d, %d) heading = %f rad %f deg  computed to point (%d, %d)\n", x, z, heading, 180 * heading / CUDART_PI_F, (int)waypoints[pos+1].x, (int)waypoints[pos+1].y);
    // }

    if (!__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, heading, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        waypoints[pos].w = 0;
}

__global__ static void __CUDA_KERNEL_SetGoalVectorized(float3 *frame, int *params, int *classCost)
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
    float angle = 0;

    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, angle, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_0;

    angle = CUDART_PI_F / 2; // 90
    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, angle, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_90;

    angle = CUDART_PI_F / 4; // 45
    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, angle, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_45;

    angle = -CUDART_PI_F / 4; // -45
    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, angle, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_MINUS_45;

    angle = 2 * CUDART_PI_F - atan2f(-dz, dx);
    // if (x == DEBUG_X && z == DEBUG_Z)
    // {
    //     printf("angle = %f\n", angle);
    // }
    if (__CUDA_KERNEL_ComputeFeasibleForAngle(frame, classCost, x, z, angle, width, height, min_dist_x, min_dist_z, lower_bound_ego_x, lower_bound_ego_z, upper_bound_ego_x, upper_bound_ego_z))
        v = v | HEADING_FROM_START;

    frame[pos].z = v;
}

uchar3 *CUDA_convertFrameColors(float3 *frame, int width, int height)
{
    const int numClasses = 29;

    uchar3 *resultImgPtr = nullptr;
    if (!cudaAllocMapped(&resultImgPtr, sizeof(uchar3) * (width * height)))
        return nullptr;

    uchar3 *classColors;
    if (!cudaAllocMapped(&classColors, sizeof(uchar3) * numClasses))
    {
        cudaFreeHost(resultImgPtr);
        return nullptr;
    }

    for (int i = 0; i < numClasses; i++)
    {
        classColors[i].x = segmentationClassColors[i][0];
        classColors[i].y = segmentationClassColors[i][1];
        classColors[i].z = segmentationClassColors[i][2];
    }

    int size = width * height;
    int numBlocks = floor(size / 256) + 1;

    __CUDA_KERNEL_FrameColor<<<numBlocks, 256>>>(frame, resultImgPtr, width, height, classColors);

    CUDA(cudaDeviceSynchronize());
    cudaFreeHost(classColors);
    return resultImgPtr;
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
    int upper_bound_z)
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

    __CUDA_KERNEL_checkFeasibleWaypoints<<<numBlocks, 256>>>(frame, params, costs, waypoints, count);

    CUDA(cudaDeviceSynchronize());
    cudaFreeHost(params);
    cudaFreeHost(costs);
    return waypoints;
    // cudaFreeHost(waypoints);
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

    __CUDA_KERNEL_SetGoalVectorized<<<numBlocks, 256>>>(frame, params, costs);

    CUDA(cudaDeviceSynchronize());
    cudaFreeHost(params);
    cudaFreeHost(costs);
}
