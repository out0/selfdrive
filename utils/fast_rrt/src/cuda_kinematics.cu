#include "cuda_basic.h"
#include "class_def.h"
#include <math_constants.h>

#define NUM_POINTS_ON_MEAN 3

__device__ static double __CUDA_KERNEL_ComputeHeading(int p1_x, int p1_y, int p2_x, int p2_y, bool *valid, int width, int height)
{
    *valid = false;
    if (p1_x == p2_x && p1_y == p2_y)
        return 0.0;

    if (p1_x < 0 || p1_y < 0 || p2_x < 0 || p2_y < 0)
        return 0.0;

    if (p1_x >= width || p1_y >= height || p2_x >= width || p2_y >= height)
        return 0.0;

    double dx = p2_x - p1_x;
    double dz = p2_y - p1_y;
    *valid = true;
    double heading = CUDART_PIO2_HI - atan2(-dz, dx);

    if (heading > CUDART_PI) // greater than 180 deg
        heading = heading - 2 * CUDART_PI;

    return heading;
}
__device__ double __CUDA_KERNEL_compute_mean_heading(double4 *waypoints, int pos, int waypoints_count, bool *valid, int width, int height)
{
    double heading = 0.0;
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
__device__ static bool __CUDA_KERNEL_ComputeFeasibleForAngle(float3 *frame, int *classCost, double *checkParams, int x, int z, double angle_radians)
{
    int width = __double2int_rn(checkParams[0]);
    int height = __double2int_rn(checkParams[1]);
    int min_dist_x = __double2int_rn(checkParams[2]);
    int min_dist_z = __double2int_rn(checkParams[3]);
    int lower_bound_ego_x = __double2int_rn(checkParams[4]);
    int lower_bound_ego_z = __double2int_rn(checkParams[5]);
    int upper_bound_ego_x = __double2int_rn(checkParams[6]);
    int upper_bound_ego_z = __double2int_rn(checkParams[7]);

    double c = cos(angle_radians);
    double s = sin(angle_radians);
    int max_size = width * height;

    for (int i = -min_dist_z; i <= min_dist_z; i++)
        for (int j = -min_dist_x; j <= min_dist_x; j++)
        {
            int xl = __double2int_rn(j * c - i * s + x);
            int zl = __double2int_rn(j * s + i * c + z);

            if (xl < 0 || xl >= width)
                continue;

            if (zl < 0 || zl >= height)
                continue;

            if (xl >= lower_bound_ego_x && xl <= upper_bound_ego_x && zl >= upper_bound_ego_z && zl <= lower_bound_ego_z)
                continue;

            int pos = zl * width + xl;
            if (pos >= max_size || pos < 0)
                continue;

            int segmentation_class = __double2int_rn(frame[pos].x);

            if (classCost[segmentation_class] < 0)
            {
                printf("[%d, %d] is not feasible on check angle because of %d, %d\n", x, z, xl, zl);
                return false;
            }
        }

    return true;
}
__device__ static double to_radians(double angle)
{
    return (angle * CUDART_PI) / 180;
}
__device__ static double to_degrees(double angle)
{
    return (angle * 180) / CUDART_PI;
}
__device__ static void convert_to_map_coord(double3 &center, double rate_w, double rate_h, double3 &p)
{
    double x = p.x;
    double z = p.y;

    p.x = (center.x - z) / rate_w;
    p.y = (x - center.y) / rate_h;
}
__device__ static void convert_to_waypoint_coord(double3 &center, double rate_w, double rate_h, double3 &p)
{
    double x = p.x;
    double y = p.y;

    p.x = __double2int_rn(center.y + rate_h * y);
    p.y = __double2int_rn(center.x - rate_w * x);
}

__device__ static double compute_euclidean_dist(double3 &start, double3 &end)
{
    double dx = end.x - start.x;
    double dy = end.y - start.y;
    return sqrt(dx * dx + dy * dy);
}
__device__ static double compute_path_heading(double3 p1, double3 p2)
{
    double dy = p2.y - p1.y;
    double dx = p2.x - p1.x;

    if (dy >= 0 && dx > 0) // Q1
        return atan(dy / dx);
    else if (dy >= 0 && dx < 0) // Q2
        return CUDART_PI - atan(dy / abs(dx));
    else if (dy < 0 && dx > 0) // Q3
        return -atan(abs(dy) / dx);
    else if (dy < 0 && dx < 0) // Q4
        return atan(dy / dx) - CUDART_PI;
    else if (dx == 0 && dy > 0)
        return CUDART_PIO2_HI;
    else if (dx == 0 && dy < 0)
        return -CUDART_PIO2_HI;
    return 0.0;
}
__device__ static double clip(double val, double min, double max)
{
    if (val < min)
        return min;
    if (val > max)
        return max;
    return val;
}

__device__ bool check_kinematic_path(float3 *og, int *classCost, double *checkParams, double3 start, double3 end)
{
    // bool DEBUG = (__double2int_rn(start.x) == 128 && __double2int_rn(start.y) == 108);

    double distance = compute_euclidean_dist(start, end);

    double3 _center;
    _center.x = checkParams[13];
    _center.y = checkParams[14];
    double _rate_w = checkParams[8];
    double _rate_h = checkParams[9];
    double _max_steering_angle_deg = checkParams[10];
    double _lr = checkParams[11];
    double velocity_meters_per_s = checkParams[12];

    convert_to_map_coord(_center, _rate_w, _rate_h, start);
    convert_to_map_coord(_center, _rate_w, _rate_h, end);
    double dt = 0.1;

    double max_turning_angle = to_radians(_max_steering_angle_deg);
    double heading = to_radians(start.z);

    double path_heading = compute_path_heading(start, end);
    double steering_angle_deg = clip(path_heading - heading, -max_turning_angle, max_turning_angle);
    double ds = velocity_meters_per_s * dt;

    int total_steps = __double2int_rn(round(distance / ds));

    double best_end_dist = distance;
    double x = start.x;
    double y = start.y;

    double3 last_p;
    double3 next_p;

    for (int i = 0; i < total_steps; i++)
    {
        double steer = tan(steering_angle_deg);
        double beta = atan(steer / _lr);

        x += ds * cos(heading + beta);
        y += ds * sin(heading + beta);
        heading += ds * cos(beta) * steer / (2 * _lr);

        next_p.x = x;
        next_p.y = y;
        next_p.z = heading;

        path_heading = compute_path_heading(next_p, end);
        steering_angle_deg = clip(path_heading - heading, -max_turning_angle, max_turning_angle);
        double dist = compute_euclidean_dist(next_p, end);

        convert_to_waypoint_coord(_center, _rate_w, _rate_h, next_p);

        if (next_p.x == last_p.x && next_p.y == last_p.z)
            continue;

        if (best_end_dist < dist)
        {
            return best_end_dist <= 2;
        }

        if (!__CUDA_KERNEL_ComputeFeasibleForAngle(og, classCost, checkParams, next_p.x, next_p.y, next_p.z))
        {
            // if (DEBUG) {
            //     //convert_to_waypoint_coord(_center, _rate_w, _rate_h, next_p);
            //     //printf("Not feasible on x,y = %d, %d, %f\n", __double2int_rd(next_p.x), __double2int_rd(next_p.y), to_degrees(next_p.z));
            // }
            return false;
        }

        last_p.x = next_p.x;
        last_p.z = next_p.z;

        best_end_dist = dist;
    }

    return false;
}

__global__ void CUDA_KERNEL_check_connection_feasible(float3 *og, int *classCost, double *checkParams, unsigned int *pcount, double3 &start, double3 &end)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = __double2int_rn(checkParams[0]);
    int height = __double2int_rn(checkParams[1]);

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (__double2int_rd(start.x) != x || __double2int_rd(start.z) != z)
        return;

    bool res = check_kinematic_path(og, classCost, checkParams, start, end);
    if (res)
        return;

    atomicInc(pcount, width * height);
}

bool CUDA_check_connection_feasible(float3 *og, int *classCost, double *checkParams, unsigned int *pcount, double3 &start, double3 &end)
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

__global__ void CUDA_KERNEL_build_path(double4 *graph, float3 *og, double *checkParams, double3 *inputParams)
{
    int width = checkParams[0];
    int height = checkParams[1];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int zt = pos / width;
    int xt = pos - zt * width;

    if (graph[pos].w != 1.0)
        return;

    double3 start = inputParams[0];
    double3 end = inputParams[1];
    int r, g, b;
    r = __double2int_rd(inputParams[2].x);
    g = __double2int_rd(inputParams[2].y);
    b = __double2int_rd(inputParams[2].z);

    // limits the process to a single thread. I dont want a bunch of threads doing the same thing...
    if (xt != __double2int_rn(start.x) || zt != __double2int_rn(start.y))
        return;

    // Now lets build the path
    double distance = compute_euclidean_dist(start, end);

    double3 _center;
    _center.x = checkParams[13];
    _center.y = checkParams[14];
    double _rate_w = checkParams[8];
    double _rate_h = checkParams[9];
    double _max_steering_angle_deg = checkParams[10];
    double _lr = checkParams[11];
    double velocity_meters_per_s = checkParams[12];

    convert_to_map_coord(_center, _rate_w, _rate_h, start);
    convert_to_map_coord(_center, _rate_w, _rate_h, end);
    double dt = 0.1;

    double max_turning_angle = to_radians(_max_steering_angle_deg);
    double heading = to_radians(start.z);

    double path_heading = compute_path_heading(start, end);
    double steering_angle_deg = clip(path_heading - heading, -max_turning_angle, max_turning_angle);
    double ds = velocity_meters_per_s * dt;

    int total_steps = __double2int_rn(distance / ds);

    double best_end_dist = distance;
    double x = start.x;
    double y = start.y;

    double3 last_p;
    double3 next_p;

    double iL = 1 / (2 * _lr);

    for (int i = 0; i < total_steps; i++)
    {
        double steer = tan(steering_angle_deg);
        double beta = atan(steer / _lr);

        x += ds * cos(heading + beta);
        y += ds * sin(heading + beta);
        heading += ds * cos(beta) * steer * iL;

        next_p.x = x;
        next_p.y = y;
        next_p.z = to_degrees(heading);

        path_heading = compute_path_heading(next_p, end);
        steering_angle_deg = clip(path_heading - heading, -max_turning_angle, max_turning_angle);
        double dist = compute_euclidean_dist(next_p, end);

        convert_to_waypoint_coord(_center, _rate_w, _rate_h, next_p);

        if (next_p.x == last_p.x && next_p.y == last_p.y)
            continue;

        if (best_end_dist < dist)
            return;

        int pos = next_p.y * width + next_p.x;

        if (pos < width * height && pos >= 0)
        {
            og[pos].x = r;
            og[pos].y = g;
            og[pos].z = b;
        }

        last_p.x = next_p.x;
        last_p.y = next_p.y;
        best_end_dist = dist;
    }
}

__device__ void change_color(float3 *og, int width, int height, int r, int g, int b, int x, int z)
{
    int p = z * width + x;
    if (p >= width * height || p < 0)
        return;
    og[p].x = r;
    og[p].y = g;
    og[p].z = b;
}

__global__ void CUDA_KERNEL_draw_nodes(double4 *graph, float3 *og, int width, int height, int r, int g, int b)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].w == 1.0)
    {
        for (int i = -1; i <= 1; i++)
        {
            change_color(og, width, height, r, g, b, x + i, z);
            change_color(og, width, height, r, g, b, x, z + i);
        }
    }
}

__global__ void __CUDA_KERNEL_find_nearest_feasible_neighbor_dist(double4 *graph, float3 *og, int *classCost, double *checkParams, int target_x, int target_z, long long *bestDistance)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = __double2int_rn(checkParams[0]);
    int height = __double2int_rn(checkParams[1]);

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].w != 1.0) // w means that the point is part of the graph
        return;

    // printf("[CUDA debug] (%d, %d) in graph\n", x, z);

    if (target_x == x && target_z == z)
    {
        atomicMin(bestDistance, 0);
        return;
    }

    int dx = target_x - x;
    int dz = target_z - z;

    // may be optimized with a max distance to check?
    long long dist = dx * dx + dz * dz;

    // printf("[CUDA debug] (%d, %d) dist = %d\n", x, z, dist);

    double3 start, end;

    start.x = __int2double_rn(x);
    start.y = __int2double_rn(z);
    start.z = graph[pos].z;

    end.x = __int2double_rn(target_x);
    end.y = __int2double_rn(target_z);
    end.z = 0.0; // NOT USED

    // printf("[CUDA debug] (%d, %d) checking %f, %f, %f -> %f, %f\n", x, z, start.x, start.y, start.z, end.x, end.y);

    if (!check_kinematic_path(og, classCost, checkParams, start, end))
    {
        // printf("[CUDA debug] (%d, %d) not feasible for %f, %f, %f -> %f, %f\n", x, z, start.x, start.y, start.z, end.x, end.y);
        return;
    }

    atomicMin(bestDistance, dist);
}

/*
This method is much less efficient than it's CPU equivalent. It is only used for testing.
*/
void __tst_CUDA_build_path(double4 *graph, float3 *og, double *checkParams, double3 &start, double3 &end, int r, int g, int b)
{
    int width = static_cast<int>(round(checkParams[0]));
    int height = static_cast<int>(round(checkParams[1]));
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    double3 *inputParams = nullptr;
    if (!cudaAllocMapped(&inputParams, sizeof(double3) * 3))
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

void __tst_CUDA_draw_nodes(double4 *graph, float3 *og, int width, int height, int r, int g, int b)
{
    int size = width * height;

    int numBlocks = floor(size / 256) + 1;

    CUDA_KERNEL_draw_nodes<<<numBlocks, 256>>>(graph, og, width, height, r, g, b);
    CUDA(cudaDeviceSynchronize());
}

__global__ void __CUDA_KERNEL_find_feasible_lowest_neighbor_cost(double4 *graph, float3 *og, int *classCost, double *checkParams, int target_x, int target_z, float radius, long long *bestCost)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int width = __double2int_rn(checkParams[0]);
    int height = __double2int_rn(checkParams[1]);

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].w != 1.0) // w means that the point is part of the graph
        return;

    int dx = target_x - x;
    int dz = target_z - z;

    long long dist = __float2ll_rd(dx * dx + dz * dz);

    long long r = __float2ll_rd(radius * radius);

    if (dist > r)
        return;

    double3 start, end;

    start.x = x;
    start.y = z;
    start.z = graph[pos].z;
    end.x = target_x;
    end.y = target_z;

    if (!check_kinematic_path(og, classCost, checkParams, start, end))
        return;

    // self cost + dist
    long long cost = __float2ll_rd(sqrt(dist) + graph[pos].z);

    atomicMin(bestCost, cost);
}

__global__ void __CUDA_KERNEL_find_feasible_neighbor_with_cost(double4 *graph, float3 *og, int *classCost, double *checkParams, int target_x, int target_z, long long *bestCost, int3 *point)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int width = __double2int_rn(checkParams[0]);
    int height = __double2int_rn(checkParams[1]);

    if (pos > width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].w != 1.0) // w means that the point is part of the graph
        return;

    int dx = target_x - x;
    int dz = target_z - z;

    long long dist = __float2ll_rd(dx * dx + dz * dz);
    long long cost = __float2ll_rd(sqrt(dist) + graph[pos].z);

    if (cost != *bestCost)
        return;

    double3 start, end;

    start.x = x;
    start.y = z;
    start.z = graph[pos].z;
    end.x = target_x;
    end.y = target_z;

    if (!check_kinematic_path(og, classCost, checkParams, start, end))
        return;

    point->x = x;
    point->y = z;
    point->z = 1;
}

__device__ bool check_node_in_subgraph(double4 *graph, int width, int current_pos, int check_pos) {
    
    int parent_pos = current_pos;

    while (parent_pos != check_pos) {

        int parent_pos_x = graph[current_pos].x;
        int parent_pos_z = graph[current_pos].y;
        
        if (parent_pos_x < 0 || parent_pos_z < 0) 
            return false;
        
        parent_pos = width * parent_pos_z + parent_pos_x;
    }

    return true;
}

__global__ void __CUDA_KERNEL_optimizeGraphWithNode(double4 *graph, double * graph_cost, float3 *og, int *classCost, double *checkParams, int parent_candidate_x, int parent_candidate_z, float radius)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int width = __double2int_rn(checkParams[0]);
    int height = __double2int_rn(checkParams[1]);

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].w != 1.0) // w means that the point is part of the graph
        return;

    int dx = parent_candidate_x - x;
    int dz = parent_candidate_z - z;

    double dist = sqrt(dx * dx + dz * dz);
    if (dist > radius)
        return;

    double3 start, end;

    int parent_candidate_pos = parent_candidate_z * width + parent_candidate_x;
    double parent_candidate_heading = graph[parent_candidate_pos].z;

    start.x = parent_candidate_x;
    start.y = parent_candidate_z;
    start.z = parent_candidate_heading;
    end.x = x;
    end.y = z;

    double current_cost = graph_cost[pos];
    double parent_cost = graph_cost[parent_candidate_pos];
    
    if (current_cost < parent_cost + dist)
        return;

    if (check_node_in_subgraph(graph, width, pos, parent_candidate_pos))
        return;

    if (!check_kinematic_path(og, classCost, checkParams, start, end))
        return;

    // we should optimize
    graph_cost[pos] = parent_cost + dist;
    //graph[]

}