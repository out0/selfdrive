
#include <math_constants.h>
#include <string>
#include <cuda_runtime.h>
#include "../../include/cuda_params.h"
#include "../../include/math_utils.h"
#include "../../include/graph.h"
#include "../../include/waypoint.h"

extern __device__ __host__ bool set(int3 *graph, float3 *graphData, long pos, float heading, int parent_x, int parent_z, float cost, int type, bool override);
extern __device__ __host__ int2 getParentCuda(int3 *graph, long pos);
extern __device__ __host__ void setTypeCuda(int3 *graph, long pos, int type);
extern __device__ __host__ float getHeadingCuda(float3 *graphData, long pos);
extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int x, int z, float angle_radians);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getCostCuda(float3 *graphData, long pos);
extern __device__ __host__ float getFrameCostCuda(float3 *frame, float *classCost, long pos);

/// @brief Converts any map coordinate (x, y) to waypoint (x, z) assuming that location = (x = 0, y = 0, heading = 0)
/// @param center
/// @param rate_w
/// @param rate_h
/// @param coord
/// @return waypoint(x, z)
__device__ __host__ int2 convert_map_pose_to_waypoint(int2 center, float rate_w, float rate_h, double2 coord)
{
    // map to waypoint formula is:
    //
    // [x' y' 1] = [x y 1] @ TranslationMatrix(lx, ly) @ RotationMatrix(-heading) @ ResizeMatrix(rh, rw)
    // x = Xcenter + y'
    // z = Zcenter - x'
    //
    // Assuming lx, ly = 0, 0  and heading = 0,  the TranslationMatrix and RotationMatrix become identity matrix.
    //
    // [x' y' 1] = [x y 1] @ ResizeMatrix(rh, rw)
    //
    // x' = x * rh
    // y' = y * rw
    //
    // x = Xcenter + y * rw
    // z = Zcenter - x * rh

    return {
        center.x + TO_INT(coord.y * rate_w),
        center.y - TO_INT(coord.x * rate_h)};
}

/// @brief Converts any waypoint (x, z) to map coordinate (x, y) assuming that location = (x = 0, y = 0, heading = 0)
/// @param center
/// @param rate_w
/// @param rate_h
/// @param coord
/// @return waypoint(x, z)
__device__ __host__ inline double2 convert_waypoint_to_map_pose(int2 center, double inv_rate_w, double inv_rate_h, int2 coord)
{
    return {
        inv_rate_h * (center.y - coord.y),
        inv_rate_w * (coord.x - center.x)};
}

__device__ __host__ double compute_euclidean_2d_dist(double2 &start, double2 &end)
{
    double dx = end.x - start.x;
    double dy = end.y - start.y;
    return sqrt(dx * dx + dy * dy);
}

__device__ __host__ double compute_euclidean_2d_dist(int2 &start, int2 &end)
{
    double dx = end.x - start.x;
    double dy = end.y - start.y;
    return sqrt(dx * dx + dy * dy);
}

__device__ __host__ double compute_path_heading(double2 p1, double2 p2)
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
__device__ __host__ double compute_path_heading(pose p1, pose p2)
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
__device__ __host__ inline double clip(double val, double min, double max)
{
    if (val < min)
        return min;
    if (val > max)
        return max;
    return val;
}

__device__ __host__ int2 draw_kinematic_path_candidate(int3 *graph, float3 *graphData, double *physicalParams, float3 *frame, float *classCosts, int width, int height, int2 center, int2 start, float steeringAngle, float pathSize, float velocity_m_s)
{
    if (physicalParams == nullptr)
    {
        printf("[Error] the physical parameters are not set\n");
        return {-1, -1};
    }

    const double rateW = physicalParams[PHYSICAL_PARAMS_RATE_W];
    const double rateH = physicalParams[PHYSICAL_PARAMS_RATE_H];
    const double invRateW = physicalParams[PHYSICAL_PARAMS_INV_RATE_W];
    const double invRateH = physicalParams[PHYSICAL_PARAMS_INV_RATE_H];
    const double maxSteering = physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];
    const double lr = physicalParams[PHYSICAL_PARAMS_LR];

    double2 startPose = convert_waypoint_to_map_pose(center, invRateW, invRateH, start);

    if (steeringAngle > maxSteering)
        steeringAngle = maxSteering;
    else if (steeringAngle < -maxSteering)
        steeringAngle = -maxSteering;

    const float steer = tanf(steeringAngle);
    const float dt = 0.1;
    const float ds = velocity_m_s * dt;
    const float beta = atanf(steer / lr);
    const float heading_increment_factor = ds * cosf(beta) * steer / (2 * lr);

    float x = startPose.x;
    float y = startPose.y;

    int maxSize = TO_INT(pathSize) + 1;

    int size = 0;

    int last_x = start.x;
    int last_z = start.y;

    float heading = getHeadingCuda(graphData, computePos(width, start.x, start.y));
    int2 lastp;

    double parentCost = getCostCuda(graphData, computePos(width, start.x, start.y));

    while (size < maxSize)
    {
        x += ds * cosf(heading + beta);
        y += ds * sinf(heading + beta);
        heading += heading_increment_factor;

        lastp = convert_map_pose_to_waypoint(center, rateW, rateH, {x, y});

        if (lastp.x == last_x && lastp.y == last_z)
            continue;

        if (lastp.x < 0 || lastp.x >= width)
            break;

        if (lastp.y < 0 || lastp.y >= height)
            break;

        long pos = width * lastp.y + lastp.x;
        size += 1;

        if (set(graph, graphData, pos, heading, last_x, last_z, 0.0, GRAPH_TYPE_PROCESSING, false))
        {
            last_x = lastp.x;
            last_z = lastp.y;
        }
    }

    return {last_x, last_z};
}


// std::vector<Waypoint> drawKinematicPath(double *physicalParams, int width, int height, int2 center, int2 start, float start_heading, Waypoint end, float velocity_m_s)
// {
//     const double rateW = physicalParams[PHYSICAL_PARAMS_RATE_W];
//     const double rateH = physicalParams[PHYSICAL_PARAMS_RATE_H];
//     const double invRateW = physicalParams[PHYSICAL_PARAMS_INV_RATE_W];
//     const double invRateH = physicalParams[PHYSICAL_PARAMS_INV_RATE_H];
//     const double lr = physicalParams[PHYSICAL_PARAMS_LR];
//     const double maxSteeringAngle = physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];

//     int2 ep = {end.x(), end.z()};
//     double2 startM = convert_waypoint_to_map_pose(center, invRateW, invRateH, start);
//     double2 endM = convert_waypoint_to_map_pose(center, invRateW, invRateH, ep);

//     double goal_heading = end.heading().rad();
//     double heading = start_heading;

//     double dt = 0.1;
//     double ds = velocity_m_s * dt;
//     int total_steps = TO_INT(compute_euclidean_2d_dist(start, ep) / ds);

//     std::vector<Waypoint> res;

//     double x;
//     double y;

//     int2 lastWP = {start.x, start.y};

//     for (int i = 0; i < total_steps; i++)
//     {
//         double steering_error = clip(goal_heading - heading, -maxSteeringAngle, maxSteeringAngle);
//         double steer = tan(steering_error);
//         double beta = atan(steer / lr);

//         x += ds * cos(heading + beta);
//         y += ds * sin(heading + beta);
//         heading += ds * cos(beta) * steer / (2 * lr);

//         int2 wp = convert_map_pose_to_waypoint(center, rateW, rateH, {x, y});
//         if (wp.x == lastWP.x && wp.y == lastWP.y)
//             continue;
//         lastWP = wp;

//         if (wp.x < 0 || wp.x >= width) continue;
//         if (wp.y < 0 || wp.y >= height) continue;


//         res.push_back(Waypoint(wp.x, wp.y, angle::rad(heading)));
//     }

//     return res;
// }



// std::pair<float3 *, int> drawKinematicIdealPath(double *physicalParams, int width, int2 center, Waypoint goal, float velocity_m_s)
// {
//     const double rateW = physicalParams[PHYSICAL_PARAMS_RATE_W];
//     const double rateH = physicalParams[PHYSICAL_PARAMS_RATE_H];
//     const double invRateW = physicalParams[PHYSICAL_PARAMS_INV_RATE_W];
//     const double invRateH = physicalParams[PHYSICAL_PARAMS_INV_RATE_H];
//     const double lr = physicalParams[PHYSICAL_PARAMS_LR];
//     const double maxSteeringAngle = physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];

//     int2 sp = {center.x, center.y};
//     int2 ep = {goal.x(), goal.z()};

//     double2 startM = convert_waypoint_to_map_pose(center, invRateW, invRateH, sp);
//     double2 endM = convert_waypoint_to_map_pose(center, invRateW, invRateH, ep);

//     double goal_heading = goal.heading().rad();
//     double heading = 0.0;

//     double dt = 0.1;
//     double ds = velocity_m_s * dt;
//     int total_steps = TO_INT(compute_euclidean_2d_dist(sp, ep) / ds);

//     float3 *path;
//     if (!cudaAllocMapped(&path, sizeof(float3) * total_steps))
//     {
//         std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(float3) * total_steps) + std::string(" bytes for drawKinematicIdealPath\n");
//         throw msg;
//     }

//     double x;
//     double y;

//     int2 lastWP = {center.x, center.y};

//     for (int i = 0; i < total_steps; i++)
//     {
//         double steering_error = clip(goal_heading - heading, -maxSteeringAngle, maxSteeringAngle);
//         double steer = tan(steering_error);
//         double beta = atan(steer / lr);

//         x += ds * cos(heading + beta);
//         y += ds * sin(heading + beta);
//         heading += ds * cos(beta) * steer / (2 * lr);

//         int2 wp = convert_map_pose_to_waypoint(center, rateW, rateH, {x, y});
//         if (wp.x == lastWP.x && wp.y == lastWP.y)
//             continue;
//         lastWP = wp;

//         path[i].x = wp.x;
//         path[i].y = wp.y;
//         path[i].z = heading;
//     }

//     return {path, total_steps};
// }

__device__ __host__ bool checkKinematicPath(int3 *graph, float3 *graphData, float3 *frame, double *physicalParams, int *params, float *classCost, int2 center, int2 start, int2 end, float velocity_m_s, float maxSteeringAngle, double &final_heading)
{
    double distance = compute_euclidean_2d_dist(start, end);

    int width = params[FRAME_PARAM_WIDTH];

    const double rateW = physicalParams[PHYSICAL_PARAMS_RATE_W];
    const double rateH = physicalParams[PHYSICAL_PARAMS_RATE_H];
    const double invRateW = physicalParams[PHYSICAL_PARAMS_INV_RATE_W];
    const double invRateH = physicalParams[PHYSICAL_PARAMS_INV_RATE_H];
    const double lr = physicalParams[PHYSICAL_PARAMS_LR];

    double2 startM = convert_waypoint_to_map_pose(center, invRateW, invRateH, start);
    double2 endM = convert_waypoint_to_map_pose(center, invRateW, invRateH, end);
    double dt = 0.1;

    long startPos = computePos(width, start.x, start.y);
    double heading = getHeadingCuda(graphData, startPos);

    double path_heading = compute_path_heading(startM, endM);
    double steering_angle_deg = clip(path_heading - heading, -maxSteeringAngle, maxSteeringAngle);
    double ds = velocity_m_s * dt;

    int total_steps = TO_INT(distance / ds);

    double2 nextpM;
    nextpM.x = startM.x;
    nextpM.y = startM.y;
    int2 nextp, lastp = {start.x, start.y};

    double bestEndDist = distance;
    final_heading = 0;

    for (int i = 0; i < total_steps; i++)
    {
        double steer = tan(steering_angle_deg);
        double beta = atan(steer / lr);

        nextpM.x += ds * cos(heading + beta);
        nextpM.y += ds * sin(heading + beta);
        heading += ds * cos(beta) * steer / (2 * lr);

        path_heading = compute_path_heading(nextpM, endM);
        steering_angle_deg = clip(path_heading - heading, -maxSteeringAngle, maxSteeringAngle);
        double dist = compute_euclidean_2d_dist(nextpM, endM);

        nextp = convert_map_pose_to_waypoint(center, rateW, rateH, nextpM);

        if (nextp.x == lastp.x && nextp.y == lastp.y)
            continue;

        if (bestEndDist < dist)
        {
            final_heading = heading;
            return bestEndDist <= 2;
        }

        if (!__computeFeasibleForAngle(frame, params, classCost, nextp.x, nextp.y, heading))
        {
            // if (DEBUG) {
            //     //convert_to_waypoint_coord(_center, _rate_w, _rate_h, next_p);
            //     //printf("Not feasible on x,y = %d, %d, %f\n", __double2int_rd(next_p.x), __double2int_rd(next_p.y), to_degrees(next_p.z));
            // }
            return false;
        }

        lastp.x = nextp.x;
        lastp.y = nextp.y;

        bestEndDist = dist;
    }

    return false;
}

// bool CudaGraph::checkFeasibleConnection(float3 *og, int2 start, int2 end, int velocity_m_s, angle maxSteeringAngle)
// {

//     double finalHeading = 0;

//     return checkKinematicPath(
//         _frame->getCudaPtr(),
//         _frameData->getCudaPtr(),
//         og,
//         _physicalParams,
//         _searchSpaceParams,
//         _classCosts,
//         _gridCenter,
//         start,
//         end,
//         velocity_m_s,
//         maxSteeringAngle.rad(),
//         finalHeading);
// }