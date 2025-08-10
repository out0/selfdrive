#include "../../include/search_frame.h"

extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
__global__ static void __CUDA_KERNEL_bestWaypointCostForHeading(float3 *frame, int *params, float *classCost, int goal_x, int goal_z, float heading, int *bestCost);
__global__ static void __CUDA_KERNEL_bestWaypointCostForPosNegHeading(float3 *frame, int *params, float *classCost, int goal_x, int goal_z, float heading, int *bestCost);
__global__ static void __CUDA_KERNEL_findWaypointForCostAndHeading(float3 *frame, int *params, float *classCost, int *bestCost, int goal_x, int goal_z, float heading, int *waypoint);
//__device__ static float __CUDA_KERNEL_ComputeHeading_Unbound_Values(int p1_x, int p1_y, int p2_x, int p2_y, bool *valid, int width, int height);

/*
#define MAX_VAL 999999
Waypoint SearchFrame::findBestWaypoint(int goal_x, int goal_z)
{
    if (!_isGoalSet)
        throw std::string("findBestWaypoint() must be called after setGoal()\n");

    size_t size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_bestValue->get() = MAX_VAL;
    int heading = 0;

    for (; heading < 90; heading+=5) {
        __CUDA_KERNEL_bestWaypointCostForPosNegHeading<<<numBlocks, THREADS_IN_BLOCK>>>(getCudaPtr(), 
            _params->get(), _classCosts->get(), goal_x, goal_z, heading, _bestValue->get());
        CUDA(cudaDeviceSynchronize());

        if (*_bestValue->get() < MAX_VAL)
            break;
    }

    // printf("best cost = %d\n", *bestCost);
    int *waypoint;
    if (!cudaAllocMapped(&waypoint, sizeof(int) * 2))
        throw std::bad_alloc();

    waypoint[0] = -1;
    waypoint[1] = -1;

    __CUDA_KERNEL_findWaypointForCostAndHeading<<<numBlocks, THREADS_IN_BLOCK>>>(getCudaPtr(), 
        _params->get(), _classCosts->get(), _bestValue->get(), goal_x, goal_z, heading, waypoint);
    CUDA(cudaDeviceSynchronize());

    Waypoint res(waypoint[0], waypoint[1], angle::rad(static_cast<float>(heading)));
    cudaFreeHost(waypoint);
    return res;
}
Waypoint SearchFrame::findBestWaypoint(int goal_x, int goal_z, float heading)
{
    size_t size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_bestValue = 999999;

    __CUDA_KERNEL_bestWaypointCostForHeading<<<numBlocks, THREADS_IN_BLOCK>>>(getCudaPtr(), 
        _params->get(), _classCosts->get(), goal_x, goal_z, heading, _bestValue->get());
    CUDA(cudaDeviceSynchronize());

    // printf("best cost = %d\n", *bestCost);
    int *waypoint;
    if (!cudaAllocMapped(&waypoint, sizeof(int) * 2))
        throw std::bad_alloc();

    waypoint[0] = -1;
    waypoint[1] = -1;

    __CUDA_KERNEL_findWaypointForCostAndHeading<<<numBlocks, THREADS_IN_BLOCK>>>(getCudaPtr(), 
        _params->get(), _classCosts->get(), _bestValue->get(), goal_x, goal_z, heading, waypoint);
    CUDA(cudaDeviceSynchronize());
    
    Waypoint res(waypoint[0], waypoint[1], angle::rad(static_cast<float>(heading)));
    cudaFreeHost(waypoint);
    return res;
}
*/

__global__ static void __CUDA_KERNEL_bestWaypointCostForHeading(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ,  int goal_x, int goal_z, float heading, int *bestCost)
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

    long cost = (dx * dx + dz * dz) % 10000000;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (classCost[(int)frame[pos].x] < 0)
        return;

    if (!__computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, goal_x, goal_z, heading))
        return;

    atomicMin(bestCost, cost);
}

__global__ static void __CUDA_KERNEL_bestWaypointCostForPosNegHeading(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int goal_x, int goal_z, float heading, int *bestCost)
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

    long cost = (dx * dx + dz * dz) % 10000000;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (classCost[(int)frame[pos].x] < 0)
        return;

    if (heading == 0)
    {
        if (!__computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, goal_x, goal_z, heading))
            return;
    }
    else
    {
        if (!__computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, goal_x, goal_z, heading) 
            && !__computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, goal_x, goal_z, -heading))
            return;
    }

    atomicMin(bestCost, cost);
}

__global__ static void __CUDA_KERNEL_findWaypointForCostAndHeading(float3 *frame, int *params, float *classCost, int *bestCost, int minDistX, int minDistZ,  int goal_x, int goal_z, float heading, int *waypoint)
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

    int cost = (dx * dx + dz * dz) % 10000000;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    if (cost == *bestCost)
    {
        // printf ("waypoint (%d, %d) has cost = %d\n", x, z, cost);
        if (!__computeFeasibleForAngle(frame, params, classCost, minDistX, minDistZ, goal_x, goal_z, heading))
            return;

        // printf ("waypoint (%d, %d) is feasible on angle %f\n", x, z, angle);

        waypoint[0] = x;
        waypoint[1] = z;
    }

    // atomicCAS(posForCost, cost == *bestCost, pos);
}


// __device__ static float __CUDA_KERNEL_ComputeHeading_Unbound_Values(int p1_x, int p1_y, int p2_x, int p2_y, bool *valid, int width, int height)
// {
//     *valid = false;
//     if (p1_x == p2_x && p1_y == p2_y)
//         return 0.0;

//     float dx = p2_x - p1_x;
//     float dz = p2_y - p1_y;
//     *valid = true;
//     float heading = CUDART_PI_F / 2 - atan2f(-dz, dx);

//     if (heading > CUDART_PI_F) // greater than 180 deg
//         heading = heading - 2 * CUDART_PI_F;

//     return heading;
// }