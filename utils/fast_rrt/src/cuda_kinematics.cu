#include "cuda_basic.h"
#include "class_def.h"
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
__device__ static bool __CUDA_KERNEL_ComputeFeasibleForAngle(float3 *frame, int *classCost, float *checkParams, int x, int z, float angle_radians)
{
    int width = __float2int_rn(checkParams[0]);
    int height = __float2int_rn(checkParams[1]);
    int min_dist_x = __float2int_rn(checkParams[2]);
    int min_dist_z = __float2int_rn(checkParams[3]);
    int lower_bound_ego_x = __float2int_rn(checkParams[4]);
    int lower_bound_ego_z = __float2int_rn(checkParams[5]);
    int upper_bound_ego_x = __float2int_rn(checkParams[6]);
    int upper_bound_ego_z = __float2int_rn(checkParams[7]);

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
                return false;
        }

    return true;
}
__device__ static float to_radians(float angle)
{
    return (angle * CUDART_PI_F) / 180;
}
__device__ static float to_degrees(float angle)
{
    return (angle * 180) / CUDART_PI_F;
}
__device__ static void convert_to_map_coord(float3 &center, float rate_w, float rate_h, float3 &p)
{
    float x = p.x;
    float z = p.y;

    p.x = (center.x - z) / rate_w;
    p.y = (x - center.y) / rate_h;
}
__device__ static void convert_to_waypoint_coord(float3 &center, float rate_w, float rate_h, float3 &p)
{
    float x = p.x;
    float y = p.y;

    p.x = __float2int_rd(center.y + rate_h * y);
    p.y = __float2int_rd(center.x - rate_w * x);
}
__device__ static float compute_euclidean_dist(float3 &start, float3 &end)
{
    float dx = end.x - start.x;
    float dy = end.y - start.y;
    return sqrtf(dx * dx + dy * dy);
}
__device__ static float compute_path_heading(float3 p1, float3 p2)
{
    float dy = p2.y - p1.y;
    float dx = p2.x - p1.x;

    if (dy >= 0 && dx > 0) // Q1
        return atan(dy / dx);
    else if (dy >= 0 && dx < 0) // Q2
        return CUDART_PI_F - atan(dy / abs(dx));
    else if (dy < 0 && dx > 0) // Q3
        return -atan(abs(dy) / dx);
    else if (dy < 0 && dx < 0) // Q4
        return atan(dy / dx) - CUDART_PI_F;
    else if (dx == 0 && dy > 0)
        return CUDART_PI_F / 2;
    else if (dx == 0 && dy < 0)
        return -CUDART_PI_F / 2;
    return 0.0;
}
__device__ static float clip(float val, float min, float max)
{
    if (val < min)
        return min;
    if (val > max)
        return max;
    return val;
}

__device__ bool check_kinematic_path(float3 *og, int *classCost, float *checkParams, float3 &start, float3 &end)
{
    float distance = compute_euclidean_dist(start, end);


    float3 _center;
    _center.x = checkParams[13];
    _center.y = checkParams[14];
    float _rate_w = checkParams[8];
    float _rate_h = checkParams[9];
    float _max_steering_angle_deg = checkParams[10];
    float _lr = checkParams[11];
    float velocity_meters_per_s = checkParams[12];


    convert_to_map_coord(_center, _rate_w, _rate_h, start);
    convert_to_map_coord(_center, _rate_w, _rate_h, end);
    float dt = 0.1;

    float max_turning_angle = to_radians(_max_steering_angle_deg);
    float heading = to_radians(start.z);

    float path_heading = compute_path_heading(start, end);
    float steering_angle_deg = clip(path_heading - heading, -max_turning_angle, max_turning_angle);
    float ds = velocity_meters_per_s * dt;

    int total_steps = __float2int_rn(round(distance / ds));

    float best_end_dist = distance;
    float x = start.x;
    float y = start.y;

    float3 last_p;
    float3 next_p;

    for (int i = 0; i < total_steps; i++)
    {
        float steer = tan(steering_angle_deg);
        float beta = atan(steer / _lr);

        x += ds * cosf(heading + beta);
        y += ds * sinf(heading + beta);
        heading += ds * cosf(beta) * steer / (2 * _lr);

        next_p.x = x;
        next_p.y = y;
        next_p.z = heading;

        path_heading = compute_path_heading(next_p, end);
        steering_angle_deg = clip(path_heading - heading, -max_turning_angle, max_turning_angle);
        float dist = compute_euclidean_dist(next_p, end);

        convert_to_waypoint_coord(_center, _rate_w, _rate_h, next_p);

        if (next_p.x == last_p.x && next_p.y == last_p.z)
            continue;

        if (best_end_dist < dist)
        {
            return best_end_dist <= 2;
        }

        if (!__CUDA_KERNEL_ComputeFeasibleForAngle(og, classCost, checkParams, next_p.x, next_p.y, next_p.z))
            return false;

        last_p.x = next_p.x;
        last_p.z = next_p.z;

        best_end_dist = dist;
    }
    
    return false;
}

__global__ void CUDA_KERNEL_check_connection_feasible(float3 *og, int *classCost, float *checkParams, unsigned int *pcount, float3 &start, float3 &end) {
    int width = __float2int_rn(checkParams[0]);
    int height = __float2int_rn(checkParams[1]);
    bool res = check_kinematic_path(og, classCost, checkParams, start, end);

    if (res) return;

    atomicInc(pcount, width * height);
}

bool CUDA_check_connection_feasible(float3 *og, int *classCost, float *checkParams, unsigned int *pcount, float3 &start, float3 &end)
{
    int width = static_cast<int>(round(checkParams[0]));
    int height = static_cast<int>(round(checkParams[1]));
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    *pcount = 0;
    CUDA_KERNEL_check_connection_feasible<<<numBlocks, 256>>>(og, classCost, checkParams, pcount, start, end);
    CUDA(cudaDeviceSynchronize());

    return *pcount == 0;

}


__global__ void CUDA_KERNEL_build_path(float4 * graph, float3 *og, float *checkParams, float3 *inputParams)
{
    int width = checkParams[0];
    int height = checkParams[1];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    int zt = pos / width;
    int xt = pos - zt * width;

    if (graph[pos].w != 1.0) return;

    float3 start = inputParams[0];
    float3 end = inputParams[1];
    int r,g, b;
    r = __float2int_rd(inputParams[2].x);
    g = __float2int_rd(inputParams[2].y);
    b = __float2int_rd(inputParams[2].z);


    // limits the process to a single thread. I dont want a bunch of threads doing the same thing...
    if (xt != __float2int_rd(start.x) || zt != __float2int_rd(start.y)) 
        return;

    // Now lets build the path
    float distance = compute_euclidean_dist(start, end);   

    float3 _center;
    _center.x = checkParams[13];
    _center.y = checkParams[14];
    float _rate_w = checkParams[8];
    float _rate_h = checkParams[9];
    float _max_steering_angle_deg = checkParams[10];
    float _lr = checkParams[11];
    float velocity_meters_per_s = checkParams[12];

    convert_to_map_coord(_center, _rate_w, _rate_h, start);
    convert_to_map_coord(_center, _rate_w, _rate_h, end);
    float dt = 0.1;

    float max_turning_angle = to_radians(_max_steering_angle_deg);
    float heading = to_radians(start.z);

    float path_heading = compute_path_heading(start, end);
    float steering_angle_deg = clip(path_heading - heading, -max_turning_angle, max_turning_angle);
    float ds = velocity_meters_per_s * dt;

    int total_steps = __float2int_rn(distance / ds);

    float best_end_dist = distance;
    float x = start.x;
    float y = start.y;

    float3 last_p;
    float3 next_p;
    printf("[CUDA] ds=%f\n", ds);

    for (int i = 0; i < total_steps; i++)
    {
        float steer = tan(steering_angle_deg);
        float beta = atan(steer / _lr);

        x += ds * cosf(heading + beta);
        y += ds * sinf(heading + beta);
        heading += ds * cosf(beta) * steer / (2 * _lr);

        if (i >= 9 && i < 20) {
            printf("[CUDA %d] x = %f, y=%f, heading=%f\n", i, x, y, heading);
        }

        next_p.x = x;
        next_p.y = y;
        next_p.z = to_degrees(heading);

        path_heading = compute_path_heading(next_p, end);
        steering_angle_deg = clip(path_heading - heading, -max_turning_angle, max_turning_angle);
        float dist = compute_euclidean_dist(next_p, end);

        convert_to_waypoint_coord(_center, _rate_w, _rate_h, next_p);

        if (next_p.x == last_p.x && next_p.y == last_p.y)
            continue;

        if (best_end_dist < dist)
            return;

        int pos = next_p.y * width + next_p.x;

        og[pos].x = r;
        og[pos].y = g;
        og[pos].z = b;

        last_p.x = next_p.x;
        last_p.y = next_p.y;
        best_end_dist = dist;
    }
}

/*
This method is much less efficient than it's CPU equivalent. It is only used for testing.
*/
void __tst_CUDA_build_path(float4 *graph, float3 *og, float *checkParams, float3 &start, float3 &end, int r, int g, int b)
{
    int width = static_cast<int>(round(checkParams[0]));
    int height = static_cast<int>(round(checkParams[1]));
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    float3 *inputParams = nullptr;
    if (!cudaAllocMapped(&inputParams, sizeof(float3) * 3))
        return;

    inputParams[0].x = start.x;
    inputParams[0].y = start.y;
    inputParams[0].z = start.z;
    inputParams[1].x = end.x;
    inputParams[1].y = end.y;
    inputParams[1].z = end.z;
    inputParams[2].x = r;
    inputParams[2].y = g;
    inputParams[2].z = b;
    

    CUDA_KERNEL_build_path<<<numBlocks, 256>>>(graph, og, checkParams, inputParams);
    CUDA(cudaDeviceSynchronize());
}