#include "../include/gpd.h"

#define MAX_VAL 99999999
#define PENALTY_COST 2000

__device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians)
{
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    float c = cosf(angle_radians);
    float s = sinf(angle_radians);

    // if (x == 0 && z == 27) {
    //     printf ("testing %d, %d  minDistX = %d, minDistZ = %d\n", x, z, minDistX, minDistZ);
    // }

    for (int i = -minDistZ; i <= minDistZ; i++)
        for (int j = -minDistX; j <= minDistX; j++)
        {
            int xl = TO_INT(j * c - i * s + x);
            int zl = TO_INT(j * s + i * c + z);

            if (xl < 0 || xl >= width)
                continue;

            if (zl < 0 || zl >= height)
                continue;

            if (xl >= lower_bound_ego_x && xl <= upper_bound_ego_x && zl >= upper_bound_ego_z && zl <= lower_bound_ego_z)
                continue;

            //  if (x == 0 && z == 27) {
            //         printf ("%d, %d\n", xl, zl);
            //      }


            int segmentation_class = TO_INT(frame[zl * width + xl].x);

            if (classCost[segmentation_class] < 0) {

                //  if (x == 0 && z == 27) {
                //     printf ("%d, %d unfeasible at %d, %d\n", x, z, xl, zl);
                //  }
                return false;
            }
        }
    return true;
}
__global__ void __CUDA_lowest_cost_for_heading(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int goal_x, int goal_z, float heading, int *bestCost)
{
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    int dx = goal_x - x;
    int dz = goal_z - z;

    int cost = TO_INT(sqrtf(dx * dx + dz * dz));

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    // if (classCost[(int)frame[pos].x] < 0 || frame[pos].y == 1.0)
    //     return;
    if (classCost[(int)frame[pos].x] < 0)
        return;

    if (((int)frame[pos].z & 0x200) > 0) {
        cost += PENALTY_COST;
    }

    bool traversableInAllAngles = ((int)frame[pos].z & 0x100) > 0;

    if (traversableInAllAngles || __computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, x, z, heading)) {
        atomicMin(bestCost, cost);
    }
    
}
__global__ void __CUDA_waypoing_with_given_cost_and_heading(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int goal_x, int goal_z, float heading, int *bestCost, int *waypoint)
{
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    int dx = goal_x - x;
    int dz = goal_z - z;

    int cost = TO_INT(sqrtf(dx * dx + dz * dz));
    
    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (classCost[(int)frame[pos].x] < 0)
        return;

    bool traversableInAllAngles = ((int)frame[pos].z & 0x100) > 0;

    if (((int)frame[pos].z & 0x200) > 0) {
        cost += PENALTY_COST;
    }

    if (cost == *bestCost)
    {
        // printf ("waypoint (%d, %d) has cost = %d\n", x, z, cost);
        if (traversableInAllAngles || __computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, x, z, heading)) {
            waypoint[0] = x;
            waypoint[1] = z;
        }
    }

    // atomicCAS(posForCost, cost == *bestCost, pos);
}
Waypoint GoalPointDiscover::findLowestCostWaypointWithHeading(SearchFrame &frame, std::pair<int, int> minDist, int goal_x, int goal_z, float heading)
{
    size_t size = frame.width() * frame.height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_bestValue->get() = MAX_VAL;

    int mx = TO_INT(minDist.first / 2);
    int mz = TO_INT(minDist.second / 2);

    __CUDA_lowest_cost_for_heading<<<numBlocks, THREADS_IN_BLOCK>>>(frame.getCudaPtr(),
                                                                    frame.getCudaFrameParamsPtr(), 
                                                                    frame.getCudaClassCostsPtr(), 
                                                                    mx, mz, goal_x, goal_z, heading, _bestValue->get());
    CUDA(cudaDeviceSynchronize());

    if (*_bestValue->get() >= MAX_VAL)
        return Waypoint(-1, -1, angle::rad(0));

    CudaPtr<int> ptr(2);
    int *waypoint = ptr.get();
    waypoint[0] = -1;
    waypoint[1] = -1;

    __CUDA_waypoing_with_given_cost_and_heading<<<numBlocks, THREADS_IN_BLOCK>>>(frame.getCudaPtr(),
                                                                                 frame.getCudaFrameParamsPtr(), 
                                                                                 frame.getCudaClassCostsPtr(), 
                                                                                 mx, mz, goal_x, goal_z, heading, _bestValue->get(), waypoint);
    CUDA(cudaDeviceSynchronize());

    Waypoint res(waypoint[0], waypoint[1], angle::rad(static_cast<float>(heading)));
    cudaFreeHost(waypoint);
    return res;
}

//----------------------------------------------------------------------------------------

__device__ float __CUDA_compute_heading_unbound(int p1_x, int p1_y, int p2_x, int p2_y, bool *valid, int width, int height)
{
    *valid = false;
    if (p1_x == p2_x && p1_y == p2_y)
        return 0.0;

    float dx = p2_x - p1_x;
    float dz = p2_y - p1_y;
    *valid = true;
    float heading = HALF_PI - atan2f(-dz, dx);

    if (heading > CUDART_PI_F) // greater than 180 deg
        heading = heading - 2 * CUDART_PI_F;

    return heading;
}
__device__ int compute_waypoint_cost(int x, int z, float current_heading, int goal_x, int goal_z, float next_heading)
{
    int dx = goal_x - x;
    int dz = goal_z - z;

    int cost = TO_INT(sqrtf(dx * dx + dz * dz)) % 10000000 + 60*abs(current_heading - next_heading);

    /*if (goal_z < 0)
        cost = (1 + z) * cost;*/

    return cost;
}
__global__ static void __CUDA_lowest_cost_reachable_waypoint_to_goal_step1(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int goalX, int goalZ, float nextHeading, int *bestCost)
{
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;
    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (classCost[(int)frame[pos].x] < 0)
        return;

    bool valid = false;
    float current_heading = __CUDA_compute_heading_unbound(x, z, goalX, goalZ, &valid, width, height);

    if (!valid)
        return;

    if (!__computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, x, z, current_heading))
        return;

    int cost = compute_waypoint_cost(x, z, current_heading, goalX, goalZ, nextHeading);
    if (((int)frame[pos].z & 0x200) > 0) {
        cost += PENALTY_COST;
    }

    atomicMin(bestCost, cost);
}
__global__ static void __CUDA_lowest_cost_reachable_waypoint_to_goal_step2(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int goalX, int goalZ, float next_heading, int *bestCost, float *waypoint)
{
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;
    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (classCost[(int)frame[pos].x] < 0)
        return;

    bool valid = false;
    float current_heading = __CUDA_compute_heading_unbound(x, z, goalX, goalZ, &valid, width, height);

    if (!valid)
        return;


    int cost = compute_waypoint_cost(x, z, current_heading, goalX, goalZ, next_heading);
    if (((int)frame[pos].z & 0x200) > 0) {
        cost += PENALTY_COST;
    }

    if (cost != *bestCost)
        return;


    if (!__computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, x, z, current_heading))
        return;

    waypoint[0] = 0.0 + x;
    waypoint[1] = 0.0 + z;
    waypoint[2] = current_heading;
}
Waypoint GoalPointDiscover::findLowestCostWaypointToGoal(SearchFrame &frame, std::pair<int, int> minDist, int goal_x, int goal_z, float next_heading)
{
    size_t size = frame.width() * frame.height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_bestValue->get() = MAX_VAL;

    int mx = TO_INT(minDist.first / 2);
    int mz = TO_INT(minDist.second / 2);

    __CUDA_lowest_cost_reachable_waypoint_to_goal_step1<<<numBlocks, THREADS_IN_BLOCK>>>(
        frame.getCudaPtr(), frame.getCudaFrameParamsPtr(), frame.getCudaClassCostsPtr(), mx, mz, goal_x, goal_z, next_heading,_bestValue->get());
    CUDA(cudaDeviceSynchronize());

    if (*_bestValue->get() >= MAX_VAL)
        return Waypoint(-1, -1, angle::rad(0));

    CudaPtr<float> ptr(3);
    float *waypoint = ptr.get();
    waypoint[0] = -1;
    waypoint[1] = -1;
    waypoint[2] = 0.0;

    __CUDA_lowest_cost_reachable_waypoint_to_goal_step2<<<numBlocks, THREADS_IN_BLOCK>>>(frame.getCudaPtr(),
                                                                                         frame.getCudaFrameParamsPtr(), frame.getCudaClassCostsPtr(), mx, mz, goal_x, goal_z, next_heading, _bestValue->get(), waypoint);
    CUDA(cudaDeviceSynchronize());

    return Waypoint(TO_INT(waypoint[0]), TO_INT(waypoint[1]), angle::rad(waypoint[2]));
}

//----------------------------------------------------------------------------------------

__device__ int __CUDA_compute_direction(int start_x, int start_z, int goal_x, int goal_z)
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
__global__ static void __CUDA_lowest_error_reachable_waypoint_to_goal_step1(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int goalX, int goalZ, float best_heading, int *bestCost)
{
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (classCost[(int)frame[pos].x] < 0)
        return;

    int direction = __CUDA_compute_direction(x, z, goalX, goalZ);

    if (direction & BOTTOM)
        return;

    float best_local_cost = 9999999;
    bool found = false;
    int angle_start = 0;
    int angle_finish = 0;

    if (direction & LEFT)
    {
        angle_start = -90;
        angle_finish = 0;
    }
    else if (direction & RIGHT)
    {
        angle_start = 0;
        angle_finish = 90;
    }
    else
        return;

    for (int angle = angle_start; angle <= angle_finish; angle += 5)
    {
        float a = __CUDA_ToRadians(angle);

        if (__computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, x, z, a))
        {
            float headingError = abs(best_heading - a) + 1;
            int dx = goalX - x;
            int dz = goalZ - z;
            int distError = dx * dx + dz * dz;
            int err = ((int)(headingError * headingError * distError)) % 1000000;

            if (err < best_local_cost)
            {
                best_local_cost = err;
                found = true;
            }
        }
        angle += 5;
    }

    if (!found)
        return;

    int local_cost = (int)best_local_cost;
    if (((int)frame[pos].z & 0x200) > 0) {
        local_cost += PENALTY_COST;
    }


    atomicMin(bestCost, local_cost);
}

__global__ static void __CUDA_lowest_error_reachable_waypoint_to_goal_step2(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int goalX, int goalZ, float best_heading, int *bestCost, int *bestHeadingError)
{
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (classCost[(int)frame[pos].x] < 0)
        return;

    int direction = __CUDA_compute_direction(x, z, goalX, goalZ);

    if (direction & BOTTOM)
        return;

    int angle_start = 0;
    int angle_finish = 0;

    if (direction & LEFT)
    {
        angle_start = -90;
        angle_finish = 0;
    }
    else if (direction & RIGHT)
    {
        angle_start = 0;
        angle_finish = 90;
    }
    else
        return;

    for (int angle = angle_start; angle <= angle_finish; angle += 5)
    {
        float a = __CUDA_ToRadians(angle);

        float headingError = abs(best_heading - a) + 1;
        int dx = goalX - x;
        int dz = goalZ - z;
        int distError = dx * dx + dz * dz;
        int err = ((int)(headingError * headingError * distError)) % 1000000;

        if (((int)frame[pos].z & 0x200) > 0) {
            err += PENALTY_COST;
        }
        if (err != *bestCost)
            continue;

        if (__computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, x, z, a))
        {
            atomicMin(bestHeadingError, 1000*headingError);
        }
        angle += 5;
    }

}

__global__ static void __CUDA_lowest_error_reachable_waypoint_to_goal_step3(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int goalX, int goalZ, float best_heading, int *bestCost, int *bestHeadingError, float *waypoint)
{
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (classCost[(int)frame[pos].x] < 0)
        return;

    int direction = __CUDA_compute_direction(x, z, goalX, goalZ);

    if (direction & BOTTOM)
        return;

    int angle_start = 0;
    int angle_finish = 0;

    if (direction & LEFT)
    {
        angle_start = -90;
        angle_finish = 0;
    }
    else if (direction & RIGHT)
    {
        angle_start = 0;
        angle_finish = 90;
    }
    else
        return;

    for (int angle = angle_start; angle <= angle_finish; angle += 5)
    {
        float a = __CUDA_ToRadians(angle);

        float headingError = abs(best_heading - a) + 1;
        int dx = goalX - x;
        int dz = goalZ - z;
        int distError = dx * dx + dz * dz;
        int err = ((int)(headingError * headingError * distError)) % 1000000;
        if (((int)frame[pos].z & 0x200) > 0) {
            err += PENALTY_COST;
        }
        if (err != *bestCost)
            continue;
        
        if (1000*headingError != *bestHeadingError)
            continue;

        if (__computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, x, z, a))
        {
            waypoint[0] = 0.0 + x;
            waypoint[1] = 0.0 + z;
            waypoint[2] = a;
            return;
        }
        angle += 5;
    }

}
Waypoint GoalPointDiscover::findLowestErrorWaypointToGoal(SearchFrame &frame, std::pair<int, int> minDist, int goal_x, int goal_z, float best_heading)
{
    size_t size = frame.width() * frame.height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_bestValue->get() = MAX_VAL;

    int mx = TO_INT(minDist.first / 2);
    int mz = TO_INT(minDist.second / 2);

    __CUDA_lowest_error_reachable_waypoint_to_goal_step1<<<numBlocks, THREADS_IN_BLOCK>>>(
        frame.getCudaPtr(), frame.getCudaFrameParamsPtr(), frame.getCudaClassCostsPtr(), mx, mz, goal_x, goal_z, best_heading, _bestValue->get());
    CUDA(cudaDeviceSynchronize());

    if (*_bestValue->get() >= MAX_VAL)
        return Waypoint(-1, -1, angle::rad(0));

    CudaPtr<int> bestHeadingError(1);
    *(bestHeadingError.get()) = MAX_VAL;

    __CUDA_lowest_error_reachable_waypoint_to_goal_step2<<<numBlocks, THREADS_IN_BLOCK>>>(frame.getCudaPtr(),
                                                                                          frame.getCudaFrameParamsPtr(), frame.getCudaClassCostsPtr(), mx, mz, goal_x, goal_z, best_heading, _bestValue->get(), bestHeadingError.get());
    CUDA(cudaDeviceSynchronize());

    if (*bestHeadingError.get() >= MAX_VAL)
        return Waypoint(-1, -1, angle::rad(0));

    CudaPtr<float> ptr(3);
    float *waypoint = ptr.get();
    waypoint[0] = -1;
    waypoint[1] = -1;
    waypoint[2] = 0.0;

    __CUDA_lowest_error_reachable_waypoint_to_goal_step3<<<numBlocks, THREADS_IN_BLOCK>>>(frame.getCudaPtr(),
                                                                                          frame.getCudaFrameParamsPtr(), frame.getCudaClassCostsPtr(), mx, mz, goal_x, goal_z, best_heading, _bestValue->get(), bestHeadingError.get(), waypoint);
    CUDA(cudaDeviceSynchronize());

    return Waypoint(TO_INT(waypoint[0]), TO_INT(waypoint[1]), angle::rad(waypoint[2]));
}

//----------------------------------------------------------------------------------------

GoalPointDiscover::GoalPointDiscover(bool computeExclusionZone)
{
    _bestValue = std::make_unique<CudaPtr<int>>(1);
    _computeExclusionZone = computeExclusionZone;
}
