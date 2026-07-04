#include "../../include/search_frame.h"
#include "../../include/cuda_basic.h"
#include "../../include/math_utils.h"
#include "../search_frame_params.h"

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

    // if (x == 108 && z == 46)
    // {
    //     printf("minDistX: %d, minDistZ: %d\n", minDistX, minDistZ);
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

            int segmentation_class = TO_INT(frame[zl * width + xl].x);

            if (classCost[segmentation_class] < 0)
            {
                // printf("(%d, %d) invalid on %d, %d segmentation_class: %d (x param = %f) class cost %f\n", x, z, xl, zl, segmentation_class, frame[zl * width + xl].x, classCost[segmentation_class]);
                return false;
            }
        }
    return true;
}

__device__ __host__ bool checkStateFeasible(float3 *searchFrame, int *params, float *classCosts, float4 *waypoints, int waypoints_size, int current_pos_waypoints, int minDistX, int minDistZ)
{
    float heading;

    const int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    const int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    const int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    const int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    int x = waypoints[current_pos_waypoints].x;
    int z = waypoints[current_pos_waypoints].y;

    waypoints[current_pos_waypoints].w = 1.0;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
    {
        return true;
    }

    heading = waypoints[current_pos_waypoints].z;
    bool res = __computeFeasibleForAngle(searchFrame, params, classCosts, minDistX, minDistZ, x, z, heading);

    if (!res)
        waypoints[current_pos_waypoints].w = 0.0;

    return res;
}

std::pair<int, int> __checkTraversableAngleBitPairCheck(float heading_rad)
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

__device__ __host__ bool CHECK_OUT_BOUNDARIES(int width, int height, int x, int z)
{
    return x < 0 || x >= width || z < 0 || z >= height;
}

__device__ __host__ void setObstacle(float3 *frame, int width, int height, int x, int z)
{
    if (CHECK_OUT_BOUNDARIES(width, height, x, z))
        return;

    const int pos = COMPUTE_POS(width, x, z);
    frame[pos].z = 1.0f;
}

__device__ __host__ bool isObstacle(float3 *frame, float *classCosts, int width, int height, int x, int z)
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

__device__ __host__ void propagateMinDistance(float3 *frame, float *classCosts, const int width, const int height, const int minDistance, int pos, int x, int z)
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
