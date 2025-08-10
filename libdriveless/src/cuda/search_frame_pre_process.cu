#include "../../include/search_frame.h"
#include "../../include/cuda_basic.h"
#include <stdexcept>

extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);

__device__ const float TRAVERSABILITY_ANGLES[] = {
    ANGLE_HEADING_MINUS_67_5,
    ANGLE_HEADING_MINUS_45,
    ANGLE_HEADING_MINUS_22_5,
    ANGLE_HEADING_0,
    ANGLE_HEADING_22_5,
    ANGLE_HEADING_45,
    ANGLE_HEADING_67_5,
    ANGLE_HEADING_90,
};

const float H_TRAVERSABILITY_ANGLES[] = {
    ANGLE_HEADING_MINUS_67_5,
    ANGLE_HEADING_MINUS_45,
    ANGLE_HEADING_MINUS_22_5,
    ANGLE_HEADING_0,
    ANGLE_HEADING_22_5,
    ANGLE_HEADING_45,
    ANGLE_HEADING_67_5,
    ANGLE_HEADING_90,
};

__device__ const int TRAVERSABILITY_BITS[] = {
    BIT_HEADING_MINUS_67_5,
    BIT_HEADING_MINUS_45,
    BIT_HEADING_MINUS_22_5,
    BIT_HEADING_0,
    BIT_HEADING_22_5,
    BIT_HEADING_45,
    BIT_HEADING_67_5,
    BIT_HEADING_90};

const int H_TRAVERSABILITY_BITS[] = {
    BIT_HEADING_MINUS_67_5,
    BIT_HEADING_MINUS_45,
    BIT_HEADING_MINUS_22_5,
    BIT_HEADING_0,
    BIT_HEADING_22_5,
    BIT_HEADING_45,
    BIT_HEADING_67_5,
    BIT_HEADING_90};

#define EIGHT_OVER_PI 20.371832716

std::pair<int, int> SearchFrame::checkTraversableAngleBitPairCheck(float heading_rad)
{

    float a = heading_rad;
    if (a > HALF_PI)
        a = a - PI;
    else if (a < -HALF_PI)
        a = a + PI;

    int p1 = TO_INT(EIGHT_OVER_PI * a) + 3;

    if (__TOLERANCE_EQUALITY(a, H_TRAVERSABILITY_ANGLES[p1]))
        return {H_TRAVERSABILITY_BITS[p1], -1};

    return {H_TRAVERSABILITY_BITS[p1], H_TRAVERSABILITY_BITS[p1 + 1]};
}

__device__ __host__ inline bool CHECK_OUT_BOUNDARIES(int width, int height, int x, int z)
{
    return x < 0 || x >= width || z < 0 || z >= height;
}

__device__ __host__ inline void setObstacle(float3 *frame, int width, int height, int x, int z)
{
    if (CHECK_OUT_BOUNDARIES(width, height, x, z))
        return;

    const int pos = COMPUTE_POS(width, x, z);
    frame[pos].z = 1.0f;
}

__device__ __host__ inline bool isObstacle(float3 *frame, float *classCosts, int width, int height, int x, int z)
{
    if (CHECK_OUT_BOUNDARIES(width, height, x, z))
        return false;

    const int pos = COMPUTE_POS(width, x, z);
    const int nodeClass = TO_INT(frame[pos].x);
    return classCosts[nodeClass] < 0;
}

__device__ __host__ void propagateObstacleInRegion(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start)
{
    // printf ("[CUDA] propagating from %d, %d to %d, %d\n", x_start, z_start, x_start+minDistance, z_start + minDistance);

    for (int z = z_start; z <= z_start + minDistance; z++)
    {
        for (int x = x_start; x <= x_start + minDistance; x++)
        {
            if (CHECK_OUT_BOUNDARIES(width, height, x, z))
            {
                continue;
            }

            // turns off the obstacle propagation-based traversability bit check (0001 XXXX XXXX) -> (0000 XXXX XXXX)
            // printf ("[CUDA] (%d, %d) current: %f, bit set value: %d\n", x, z, frame[COMPUTE_POS(width, x, z)].z, TO_INT(frame[COMPUTE_POS(width, x, z)].z) & 0x0FF);
            frame[COMPUTE_POS(width, x, z)].z = TO_INT(frame[COMPUTE_POS(width, x, z)].z) & 0x0FF;
        }
    }
}
__device__ __host__ void propagateObstacleLeft(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start)
{
    for (int x = x_start - minDistance; x <= x_start; x++)
    {
        if (CHECK_OUT_BOUNDARIES(width, height, x, z_start))
            continue;
        frame[COMPUTE_POS(width, x, z_start)].z = TO_INT(frame[COMPUTE_POS(width, x, z_start)].z) & 0x0FF;
    }
}
__device__ __host__ void propagateObstacleRight(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start)
{
    for (int x = x_start; x <= x_start + minDistance; x++)
    {
        if (CHECK_OUT_BOUNDARIES(width, height, x, z_start))
            continue;
        frame[COMPUTE_POS(width, x, z_start)].z = TO_INT(frame[COMPUTE_POS(width, x, z_start)].z) & 0x0FF;
    }
}
__device__ __host__ void propagateObstacleTop(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start)
{
    for (int z = z_start - minDistance; z <= z_start; z++)
    {
        if (CHECK_OUT_BOUNDARIES(width, height, x_start, z))
            continue;
        frame[COMPUTE_POS(width, x_start, z)].z = TO_INT(frame[COMPUTE_POS(width, x_start, z)].z) & 0x0FF;
    }
}
__device__ __host__ void propagateObstacleBottom(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start)
{
    for (int z = z_start; z <= z_start + minDistance; z++)
    {
        if (CHECK_OUT_BOUNDARIES(width, height, x_start, z))
            continue;
        frame[COMPUTE_POS(width, x_start, z)].z = TO_INT(frame[COMPUTE_POS(width, x_start, z)].z) & 0x0FF;
    }
}

__device__ __host__ inline void propagateMinDistance(float3 *frame, float *classCosts, const int width, const int height, const int minDistance, int pos, int x, int z)
{
    bool tl = true;
    bool tr = true;
    bool bl = true;
    bool br = true;
    bool l = true;
    bool r = true;
    bool t = true;
    bool b = true;

    if (isObstacle(frame, classCosts, width, height, x, z - 1))
    { // TOP is an obstacle
        tl = false;
        tr = false;
        t = false;
    }
    if (isObstacle(frame, classCosts, width, height, x, z + 1))
    { // BOTTOM is an obstacle
        bl = false;
        br = false;
        b = false;
    }
    if (isObstacle(frame, classCosts, width, height, x - 1, z))
    { // LEFT is an obstacle
        tl = false;
        bl = false;
        l = false;
    }
    if (isObstacle(frame, classCosts, width, height, x + 1, z))
    { // RIGHT is an obstacle
        tr = false;
        br = false;
        r = false;
    }
    if (tl & isObstacle(frame, classCosts, width, height, x - 1, z - 1))
    { // TOP left is obstacle
        tl = false;
    }
    if (tr & isObstacle(frame, classCosts, width, height, x + 1, z - 1))
    { // TOP right is obstacle
        tr = false;
    }
    if (bl & isObstacle(frame, classCosts, width, height, x - 1, z + 1))
    { // BOTTOM left is obstacle
        bl = false;
    }
    if (br & isObstacle(frame, classCosts, width, height, x + 1, z + 1))
    { // BOTTOM right is obstacle
        br = false;
    }

    // printf("[CUDA] (%d, %d) regions to propagate obstacle: tl=%d, tr=%d, bl=%d, br=%d, t=%d, b=%d, l=%d, r=%d\n", x, z, tl, tr, bl, br, t, b, l, r);

    if (tl)
        propagateObstacleInRegion(frame, width, height, minDistance, x - minDistance, z - minDistance);
    if (tr)
        propagateObstacleInRegion(frame, width, height, minDistance, x, z - minDistance);
    if (bl)
        propagateObstacleInRegion(frame, width, height, minDistance, x - minDistance, z);
    if (br)
        propagateObstacleInRegion(frame, width, height, minDistance, x, z);
    if (l && !tl)
        propagateObstacleLeft(frame, width, height, minDistance, x, z);
    if (r && !tr)
        propagateObstacleRight(frame, width, height, minDistance, x, z);
    if (t && !(tl || tr))
        propagateObstacleTop(frame, width, height, minDistance, x, z);
    if (b && !(bl || br))
        propagateObstacleBottom(frame, width, height, minDistance, x, z);
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

__global__ void __CUDA_safe_distance_obstacle_expansion_based(float3 *frame, float *classCosts, int *_searchSpaceParams, int half_minDist_px)
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

    _minDistanceChecked.first = 0.5 * minDistance.first;
    _minDistanceChecked.second = 0.5 * minDistance.second;

    int half_minDist_px = TO_INT(sqrtf(_minDistanceChecked.first * _minDistanceChecked.first + _minDistanceChecked.second * _minDistanceChecked.second));

    __CUDA_safe_distance_prepare<<<numBlocks, THREADS_IN_BLOCK>>>(getCudaPtr(), _classCosts->get(), _params->get(), half_minDist_px);
    CUDA(cudaDeviceSynchronize());

    __CUDA_safe_distance_obstacle_expansion_based<<<numBlocks, THREADS_IN_BLOCK>>>(getCudaPtr(), _classCosts->get(), _params->get(), half_minDist_px);
    CUDA(cudaDeviceSynchronize());

    _safeZoneChecked = true;

    if (computeVectorized)
    {

        __CUDA_safe_distance_vector_based<<<numBlocks, THREADS_IN_BLOCK>>>(getCudaPtr(), _classCosts->get(), _params->get(), _minDistanceChecked.first, _minDistanceChecked.second);
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

    __CUDA_distance_to_goal<<<numBlocks, THREADS_IN_BLOCK>>>(getCudaPtr(), _classCosts->get(), _params->get(), x, z);
    CUDA(cudaDeviceSynchronize());
}

float SearchFrame::getDistanceToGoal(int x, int z)
{
    float3 *ptr = getCudaPtr();
    return ptr[z * width() + x].y;
}