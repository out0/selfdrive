#include "wpmp_data.h"
#include "../include/wpmp_graph.h"
#include <driveless/math_utils.h>
#include <driveless/search_zone_utils.h>

extern __device__ __host__ float traversability_cost(float3 *frame, int *params, float *classCost, int2 min_distance, int x, int z, float angle_radians);

extern __device__ __host__ float distance(float3 p1, float3 p2);

__device__ __host__ float check_hermite_curve(float3 *frame, int *params, float *classCost, int2 min_distance, float3 p1, float3 p2,
                                              float wheelbase_px, float delta_max_rad)
{

    const int plane_width = params[FRAME_PARAM_WIDTH];
    const int plane_height = params[FRAME_PARAM_HEIGHT];

    float d = distance(p1, p2);
    float kappa_max = tanf(delta_max_rad) / wheelbase_px;

    float a1 = p1.z - PI / 2;
    float a2 = p2.z - PI / 2;

    float2 tan1 = {d * cosf(a1), d * sinf(a1)};
    float2 tan2 = {d * cosf(a2), d * sinf(a2)};

    int maxPoints = 2 * TO_INT(d);
    if (maxPoints < 2)
        return -1;

    int last_x = -1;
    int last_z = -1;

    float curve_cost = 0;

    for (int i = 0; i < maxPoints; ++i)
    {
        float t = TO_FLOAT(i) / (maxPoints - 1);
        float t2 = t * t;
        float t3 = t2 * t;

        // Position basis
        float h00 = 2 * t3 - 3 * t2 + 1;
        float h10 = t3 - 2 * t2 + t;
        float h01 = -2 * t3 + 3 * t2;
        float h11 = t3 - t2;

        float x = h00 * p1.x + h10 * tan1.x + h01 * p2.x + h11 * tan2.x;
        float z = h00 * p1.y + h10 * tan1.y + h01 * p2.y + h11 * tan2.y;

        // First derivative basis
        float h00d = 6 * t2 - 6 * t;
        float h10d = 3 * t2 - 4 * t + 1;
        float h01d = -6 * t2 + 6 * t;
        float h11d = 3 * t2 - 2 * t;

        float xp = h00d * p1.x + h10d * tan1.x + h01d * p2.x + h11d * tan2.x;
        float zp = h00d * p1.y + h10d * tan1.y + h01d * p2.y + h11d * tan2.y;

        // Second derivative basis
        float h00dd = 12 * t - 6;
        float h10dd = 6 * t - 4;
        float h01dd = -12 * t + 6;
        float h11dd = 6 * t - 2;

        float xpp = h00dd * p1.x + h10dd * tan1.x + h01dd * p2.x + h11dd * tan2.x;
        float zpp = h00dd * p1.y + h10dd * tan1.y + h01dd * p2.y + h11dd * tan2.y;

        // Curvature check — bail immediately, curve gets discarded by caller
        float denom = powf(xp * xp + zp * zp, 1.5f);
        float kappa = (denom > 1e-6f) ? fabsf(xp * zpp - zp * xpp) / denom : 0.0f;
        if (kappa > kappa_max)
            return -1;

        if (x < 0 || x >= plane_width || z < 0 || z >= plane_height)
            continue;

        int cx = TO_INT(x);
        int cz = TO_INT(z);
        if (cx == last_x && cz == last_z)
            continue;
        if (cx < 0 || cx >= plane_width || cz < 0 || cz >= plane_height)
            continue;

        float heading = atan2f(zp, xp) + HALF_PI;

        float point_cost = traversability_cost(frame, params, classCost, min_distance, cx, cz, heading);

        if (point_cost < 0)
            return -1;

        last_x = cx;
        last_z = cz;
        curve_cost += point_cost;
    }

    return curve_cost;
}

__device__ __host__ void consolidate_curve(int4 *node_conf,
                                           float4 *node_data,
                                           long pos)
{
    SET_NODE_TYPE(node_conf, pos, NODE_TYPE_GRAPH);
}

__device__ __host__ float kinematic_curve(
    float3 *frame,
    int *params,
    float *classCost,
    float steering_angle_rad,
    int wheelbase_px,
    long initial_pos,
    int4 *node_conf,
    float4 *node_data,
    int max_size_px,
    float3 goal,
    float max_steering_angle,
    float max_dist_error_px,
    float max_heading_error_rad)
{
    const float steer = tanf(steering_angle_rad);
    const float dt = 0.1;
    const float beta = atanf(steer / 2);
    const float curvature = (0.1 * cosf(beta) * steer) / (2 * wheelbase_px);
    const int width = params[FRAME_PARAM_WIDTH];
    const int height = params[FRAME_PARAM_HEIGHT];
    const int2 min_distances = {params[FRAME_PARAM_MIN_DIST_X], params[FRAME_PARAM_MIN_DIST_Z]};

    const int initial_z = initial_pos / width;
    const int initial_x = initial_pos - initial_z * width;
    const float initial_heading = NODE_HEADING(node_data, initial_pos);
    const float initial_cost = NODE_COST_FROM_START(node_data, initial_pos);

    const bool precomputed_distance_to_goal = params[FRAME_PREPROCESS_DIST_TO_GOAL_ENABLED];

    float heading = initial_heading - HALF_PI;
    float x = 0.0 + initial_x;
    float z = 0.0 + initial_z;

    int last_x = x;
    int last_z = z;

    float total_cost = initial_cost;
    int size = 0;

    while (size < max_size_px)
    {
        x += dt * cosf(heading + beta);
        z += dt * sinf(heading + beta);
        heading += curvature;

        int cx = TO_INT(x);
        int cz = TO_INT(z);

        if (cx == last_x && cz == last_z)
            continue;

        if (cx < 0 || cx >= width || cz < 0 || cz >= height)
            return -1;

        const long new_pos = COMPUTE_POS(width, cx, cz);
        const int node_type = NODE_TYPE(node_conf, new_pos);

        switch (node_type)
        {
        case NODE_TYPE_NULL:
            break;

        case NODE_TYPE_GRAPH:
            // collision with an unsolved node
            // TODO: solve this?
            continue;
            // return false;

        // case NODE_TYPE_GRAPH_TEMP:
        // case NODE_TYPE_GRAPH_TEMP_SOLUTION:
        case NODE_TYPE_ORIGIN:
            continue;
            // collision with the origin
            // return false;

        case NODE_TYPE_NULL_CONNECTED_TO_GOAL:
        {
            float3 p1 = {x, z, TO_FLOAT(heading + HALF_PI)};
            float goal_cost = check_hermite_curve(frame, params, classCost, min_distances,
                                                  p1, goal, wheelbase_px, max_steering_angle);
            if (goal_cost > 0)
            {
                SET_NODE_HEADING(node_data, new_pos, heading + HALF_PI);
                SET_NODE_PARENT(node_conf, new_pos, last_x, last_z);
                SET_NODE_TYPE(node_conf, new_pos, NODE_TYPE_GRAPH_CONNECTED_TO_GOAL);
                SET_NODE_COST_FROM_START(node_data, new_pos, total_cost);
            }
            return total_cost;
        }
        case NODE_TYPE_GRAPH_CONNECTED_TO_GOAL:
            // collision with an already solved node
            return -1;

        case NODE_TYPE_NULL_SZ_IN_CHECK:
            printf("invalid node state at %d, %d: node type is NODE_TYPE_NULL_SZ_IN_CHECK but this is a temporary state\n", cx, cz);
            return -1;
            // #ifndef DRIVELESS_CUDA_ENABLED
            //             throw std::invalid_argument("invalid node state NODE_TYPE_NULL_SZ_IN_CHECK\n");
            // #endif
            //            return false;
        case NODE_TYPE_GRAPH_SZ_IN_CHECK:
            printf("invalid node state at %d, %d: node type is NODE_TYPE_GRAPH_SZ_IN_CHECK but this is a temporary state\n", cx, cz);
            return -1;
            // #ifndef DRIVELESS_CUDA_ENABLED
            //             throw std::invalid_argument("invalid node state NODE_TYPE_GRAPH_SZ_IN_CHECK\n");
            // #endif
            //            return false;
        default:
            printf("unhandled state at %d, %d\n", cx, cz);
            return -1;
            // #ifndef DRIVELESS_CUDA_ENABLED
            //             throw std::invalid_argument("unhandled state\n");
            // #endif
            // return false;
        }

        float cost = traversability_cost(frame, params, classCost, min_distances, cx, cz, heading);
        if (cost < 0)
            return -1;

        last_x = cx;
        last_z = cz;
        size++;
        total_cost += cost;

        const float new_heading = heading + HALF_PI;

        SET_NODE_PARENT(node_conf, new_pos, last_x, last_z);
        SET_NODE_TYPE(node_conf, new_pos, NODE_TYPE_GRAPH);
        SET_NODE_HEADING(node_data, new_pos, new_heading);

        // check if we are close enough to the goal to be considered as the goal.
        if (precomputed_distance_to_goal)
        {
            if (GET_PRECOMPUTED_DISTANCE_TO_GOAL(frame, new_pos) <= max_dist_error_px //
                && abs(new_heading - goal.z) <= max_heading_error_rad)
            {
                SET_NODE_TYPE(node_conf, new_pos, NODE_TYPE_SOLUTION);
                return total_cost;
            }
        }
        else
        {
            const float dx = goal.x - TO_FLOAT(last_x);
            const float dz = goal.y - TO_FLOAT(last_x);
            const float dist = sqrtf(dx * dx + dz * dz);
            if (dist <= max_dist_error_px && abs(new_heading - goal.z) <= max_heading_error_rad)
            {
                SET_NODE_TYPE(node_conf, new_pos, NODE_TYPE_SOLUTION);
                return total_cost;
            }
        }
    }

    return -1;
}


__device__ __host__ float expand_node(float3 *frame,
                                     int *params,
                                     float *physical_params,
                                     float *class_costs,
                                     int x, int z,
                                     int4 *node_conf,
                                     float4 *node_data,
                                     int max_size_px,
                                     float3 goal,
                                     float max_dist_error_px,
                                     float max_heading_error_rad)
{
    float max_steering = physical_params[PHYSICAL_PARAM_MAX_STEERING_RAD];
    float half_max_steering = 0.5 * physical_params[PHYSICAL_PARAM_MAX_STEERING_RAD];
    float wheelbase_px = physical_params[PHYSICAL_PARAM_WHEELBASE_PX];
    const int width = params[FRAME_PARAM_WIDTH];
    long initial_pos = COMPUTE_POS(width, x, z);

    const float angles[] = {0.0, -half_max_steering, +half_max_steering, -max_steering, +max_steering};

    for (float angle : angles)
    {
        float cost = kinematic_curve(frame, params, class_costs,
                            angle, wheelbase_px,
                            initial_pos, node_conf,
                            node_data, max_size_px, goal, max_steering,
                            max_dist_error_px, max_heading_error_rad);
        if (cost >= 0)
            return cost;
    }

    return -1;
}


