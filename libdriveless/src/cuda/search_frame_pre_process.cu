#include "../../include/search_frame.h"
#include "../../include/cuda_basic.h"
#include "../search_frame_params.h"
#include <stdexcept>

extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern std::pair<int, int> __checkTraversableAngleBitPairCheck(float heading_rad);
extern __device__ __host__ bool CHECK_OUT_BOUNDARIES(int width, int height, int x, int z);
extern __device__ __host__ void setObstacle(float3 *frame, int width, int height, int x, int z);
extern __device__ __host__ bool isObstacle(float3 *frame, float *classCosts, int width, int height, int x, int z);
extern __device__ __host__ void propagateObstacleInRegion(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start);
extern __device__ __host__ void propagateObstacleLeft(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start);
extern __device__ __host__ void propagateObstacleRight(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start);
extern __device__ __host__ void propagateObstacleTop(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start);
extern __device__ __host__ void propagateObstacleBottom(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start);
extern __device__ __host__ void propagateMinDistance(float3 *frame, float *classCosts, const int width, const int height, const int minDistance, int pos, int x, int z);
extern __device__ __host__ void count_obstacle_in_search_zones(float3 *frame, float *classCosts, int *search_params, uint4 *search_zone_info, int pos);

std::pair<int, int> SearchFrame::checkTraversableAngleBitPairCheck(float heading_rad)
{
    return __checkTraversableAngleBitPairCheck(heading_rad);
}

__global__ void __CUDA_safe_distance_prepare(float3 *frame, float *classCosts, int *_searchSpaceParams, int half_minDist_px)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = _searchSpaceParams[FRAME_PARAM_WIDTH];
    int height = _searchSpaceParams[FRAME_PARAM_HEIGHT];
    int lower_bound_ego_x = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_Z];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;
    // int dx = x - goal_x;
    // int dz = z - goal_z;

    // frame[pos].y = sqrtf(dx * dx + dz * dz);

    // turns on the obstacle propagation-based traversability bit and off the angle-based bits (0001 0000 0000)
    // because the obstacle propagation-based works by propagating obstacles as turning bits off, while the
    // angle-based check works by checking each angle and turning the respective bit on as it is traversable.
    frame[pos].z = 256.0;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    const int nodeClass = TO_INT(frame[pos].x);
    if (classCosts[nodeClass] < 0)
    {
        frame[pos].z = 0x0;
    }
}

__global__ void __CUDA_safe_distance_obstacle_expansion_based(float3 *frame, float *classCosts, int *_searchSpaceParams, int half_minDist_px, int2 search_zone_dim, uint4 *search_zone_info)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int width = _searchSpaceParams[FRAME_PARAM_WIDTH];
    int height = _searchSpaceParams[FRAME_PARAM_HEIGHT];
    int lower_bound_ego_x = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_Z];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        return;

    const int nodeClass = TO_INT(frame[pos].x);

    if (classCosts[nodeClass] < 0)
    {
        // printf("[CUDA] pos %d, %d will propagate distance %d\n", x, z, half_minDist_px);
        propagateMinDistance(frame, classCosts, width, height, half_minDist_px, pos, x, z);
        count_obstacle_in_search_zones(frame, classCosts, _searchSpaceParams, search_zone_info, pos);
    }
}

__global__ void __CUDA_safe_distance_vector_based(float3 *frame, float *classCosts, int *_searchSpaceParams, int minDistX, int minDistZ)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int width = _searchSpaceParams[FRAME_PARAM_WIDTH];
    int height = _searchSpaceParams[FRAME_PARAM_HEIGHT];
    int lower_bound_ego_x = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_Z];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
    {
        frame[pos].z = 0.0 + (TO_INT(frame[pos].z) | 0xff);
        return;
    }

    const int nodeClass = TO_INT(frame[pos].x);

    if (classCosts[nodeClass] < 0)
        return;

    int v = 0;
    for (int i = 0; i < 8; i++)
    {
        if (__computeFeasibleForAngle(frame, _searchSpaceParams, classCosts, minDistX, minDistZ, x, z, TRAVERSABILITY_ANGLES[i]))
            v = v | TRAVERSABILITY_BITS[i];
    }

    frame[pos].z = 0.0 + (TO_INT(frame[pos].z) | v);
}

void SearchFrame::processSafeDistanceZone(std::pair<int, int> minDistance, bool computeVectorized)
{
    int size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    _params->get()[FRAME_PARAM_MIN_DIST_X] = 0.5 * minDistance.first;
    _params->get()[FRAME_PARAM_MIN_DIST_Z] = 0.5 * minDistance.second;

    int min_x = _params->get()[FRAME_PARAM_MIN_DIST_X];
    int min_z = _params->get()[FRAME_PARAM_MIN_DIST_Z];

    int minDist_px = TO_INT(sqrtf(min_x * min_x + min_z * min_z));

    __CUDA_safe_distance_prepare<<<numBlocks, THREADS_IN_BLOCK>>>(getPtr(), _classCosts->get(), _params->get(), minDist_px);
    CUDA(cudaDeviceSynchronize());

    _search_zone_info->clear();

    __CUDA_safe_distance_obstacle_expansion_based<<<numBlocks, THREADS_IN_BLOCK>>>(getPtr(), _classCosts->get(), _params->get(), minDist_px, {_searchZoneDim.first, _searchZoneDim.second}, _search_zone_info->getPtr());
    CUDA(cudaDeviceSynchronize());

    _safeZoneChecked = true;

    if (computeVectorized)
    {
        __CUDA_safe_distance_vector_based<<<numBlocks, THREADS_IN_BLOCK>>>(getPtr(), _classCosts->get(), _params->get(), min_x, min_z);
        CUDA(cudaDeviceSynchronize());
        _safeZoneVectorialChecked = true;
    }
}

__global__ void __CUDA_distance_to_goal(float3 *frame, float *classCosts, int *_searchSpaceParams, int goal_x, int goal_z)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int width = _searchSpaceParams[FRAME_PARAM_WIDTH];
    int height = _searchSpaceParams[FRAME_PARAM_HEIGHT];
    int lower_bound_ego_x = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_Z];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (x < lower_bound_ego_x || x > upper_bound_ego_x || z < upper_bound_ego_z || z > lower_bound_ego_z)
    {
        const int nodeClass = TO_INT(frame[pos].x);
        if (classCosts[nodeClass] < 0)
        {
            frame[pos].y = 999999999;
            return;
        }
    }

    float dx = goal_x - x;
    float dz = goal_z - z;

    frame[pos].y = sqrtf(dx * dx + dz * dz);
}

void SearchFrame::processDistanceToGoal(int x, int z)
{
    int size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    if (_classCosts->get() == nullptr)
    {
        throw std::runtime_error("Class costs were not set. Please set costs before processing distance to goal.");
    }

    __CUDA_distance_to_goal<<<numBlocks, THREADS_IN_BLOCK>>>(getPtr(), _classCosts->get(), _params->get(), x, z);
    CUDA(cudaDeviceSynchronize());
    _distanceToGoalProcessed = true;
}

float SearchFrame::getDistanceToGoal(int x, int z)
{
    float3 *ptr = getPtr();
    return ptr[z * width() + x].y;
}

std::pair<int4, float> computeICR(float *physical_params, Waypoint p1, bool invert_angle)
{
    const float max_steering_angle = physical_params[PHYSICAL_PARAM_MAX_STEERING_RAD];
    const float wheelbase_px = physical_params[PHYSICAL_PARAM_WHEELBASE_PX];
    const float steer = tanf(max_steering_angle);
    const float beta = atanf(steer / 2);
    float curvature = cosf(beta) * steer / (2 * wheelbase_px);
    if (curvature < 0)
        curvature = -1 * curvature;
    const float R = 1 / curvature;
    const float Rsq = R * R;
    const float heading = invert_angle ? p1.heading().rad() + PI : p1.heading().rad();

    int4 coordinates;

    coordinates.x = p1.x() + R * cosf(heading + beta);
    coordinates.y = p1.z() + R * sinf(heading + beta);
    coordinates.z = p1.x() - R * cosf(heading - beta);
    coordinates.w = p1.z() - R * sinf(heading - beta);
    return {coordinates, Rsq};
}

__global__ void __CUDA_process_kinematic_exclusion_areas(float3 *frame, int *_searchSpaceParams, int4 origin, int4 goal, float Rsqd)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = _searchSpaceParams[FRAME_PARAM_WIDTH];
    const int height = _searchSpaceParams[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    const int z = pos / width;
    const int x = pos - z * width;

    const int dx1 = origin.x - x;
    const int dz1 = origin.y - z;
    const int dx2 = origin.z - x;
    const int dz2 = origin.w - z;

    if ((dx1 * dx1 + dz1 * dz1) <= Rsqd) {
        frame[pos].z = 0.0;
        return;
    }
    if ((dx2 * dx2 + dz2 * dz2) <= Rsqd) {
        frame[pos].z = 0.0;
        return;
    }

    const int dx3 = goal.x - x;
    const int dz3 = goal.y - z;
    const int dx4 = goal.z - x;
    const int dz4 = goal.w - z;

    if ((dx3 * dx3 + dz3 * dz3) <= Rsqd) {
        frame[pos].z = 0.0;
        return;
    }
    if ((dx4 * dx4 + dz4 * dz4) <= Rsqd) {
        frame[pos].z = 0.0;
        return;
    }
}

void SearchFrame::processKinematicExclusionAreas(Waypoint origin, Waypoint goal)
{
    if (getPhysicalParamsPtr() == nullptr)
    {
        throw std::invalid_argument("Can only execute processKinematicExclusionAreas with physical params set\n");
    }

    int size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;


    std::pair<int4, float> icr_origin = computeICR(getPhysicalParamsPtr(), origin, false);
    std::pair<int4, float> icr_goal = computeICR(getPhysicalParamsPtr(), goal, true);

    __CUDA_process_kinematic_exclusion_areas<<<numBlocks, THREADS_IN_BLOCK>>>(getPtr(), getFrameParamsPtr(), icr_origin.first, icr_goal.first, icr_goal.second);
    CUDA(cudaDeviceSynchronize());

}
