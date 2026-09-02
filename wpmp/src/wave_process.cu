#include "wpmp_data.h"
#include <driveless/math_utils.h>
#include <driveless/search_zone_utils.h>

typedef float (*interpolation_callback)(void *, int, int, float);

typedef struct hermite_check_conf
{
    float3 *frame;
    int *params;
    float *classCost;
    int2 min_distance;
} hermite_check_conf;

extern __device__ __host__ float hermite_curve(int2 plane_dim, float3 p1, float3 p2,
                                               float wheelbase, float delta_max_rad, interpolation_callback cb, void *result_ptr);
extern __device__ __host__ float traversability_cost(float3 *frame, int *params, float *classCost, int2 min_distance, int x, int z, float angle_radians);
extern __device__ __host__ int2 zone_location(int2 zone_dim_size, int2 zone_grid_size, int x, int z);
extern __device__ __host__ bool is_zone_border(int x, int z, int xg, int zg, int search_zone_dim_w, int search_zone_dim_h);
extern __device__ __host__ bool is_zone_edge(int x, int z, int xg, int zg, int2 search_zone_dim);
extern __device__ __host__ int4 sz_egdes_frame_pos(int2 sz_location, int2 search_zone_dim, int search_zone_width);

__device__ __host__ float hermite_connection_check(void *ptr, int x, int z, float heading)
{
    auto conf = (hermite_check_conf *)ptr;
    float3 *frame = conf->frame;
    int *params = conf->params;
    float *class_costs = conf->classCost;
    int2 min_distances = conf->min_distance;
    // const int width = params[FRAME_PARAM_WIDTH];

    return traversability_cost(frame, params, class_costs, min_distances, x, z, heading);
    // if (cost < 0)
    //     return -1;

    // printf("(%d, %d) is valid with heading: %f deg;\n", x, z, (heading*180)/PI);

    // return frame[COMPUTE_POS(width, x, z)].y;
};
__device__ __host__ float compute_heading(int x, int z, float3 p2)
{
    double dz = p2.y - z;
    double dx = p2.x - x;

    if (dx == 0 && dz == 0)
        return 0;

    double v1 = 0;
    if (dz != 0)
        v1 = atan2f(-dz, dx);
    else
        v1 = atan2f(0, dx);

    return HALF_PI - v1;
}
__device__ __host__ void to_goal_wave(float3 *frame,
                                      int *params,
                                      float *physical_params,
                                      float *class_costs,
                                      int pos,
                                      float3 goal,
                                      int4 *node_conf,
                                      float4 *node_data,
                                      uint4 *search_zone_info)
{
    const int2 frame_dim = {params[FRAME_PARAM_WIDTH], params[FRAME_PARAM_HEIGHT]};
    const int2 min_distances = {params[FRAME_PARAM_MIN_DIST_X], params[FRAME_PARAM_MIN_DIST_Z]};
    const int2 zone_dim = {params[FRAME_SEARCH_ZONE_DIM_WIDTH], params[FRAME_SEARCH_ZONE_DIM_HEIGHT]};
    const int2 zone_grid_size = {params[FRAME_SEARCH_ZONE_GRID_WIDTH], params[FRAME_SEARCH_ZONE_GRID_HEIGHT]};
    const float wheelbase = physical_params[PHYSICAL_PARAM_WHEELBASE_PX];
    const float delta_max_rad = physical_params[PHYSICAL_PARAM_MAX_STEERING_RAD];

    int z = pos / frame_dim.x;
    int x = pos - z * frame_dim.x;

    int2 sz_location = zone_location(zone_dim, zone_grid_size, x, z);
    int zone_pos = SEARCH_ZONE_POS(zone_grid_size.x, sz_location.x, sz_location.y);

    const bool is_node = IS_NODE(node_conf, pos);
    float cost = -1;
    float heading = -1;

    const bool pre_process_collision_vector = params[FRAME_PREPROCESS_COLLISION_TYPE] == PREPROCESS_COLLISION_VECTOR;
    const bool pre_process_collision_dist = pre_process_collision_vector || params[FRAME_PREPROCESS_COLLISION_TYPE] == PREPROCESS_COLLISION_DIST;

    if (pre_process_collision_dist)
    {
        if (TO_INT(frame[pos].z) == 0)
            return;
    }

    if (SEARCH_ZONE_TOTAL_OBSTACLES(search_zone_info, zone_pos) == 0)
    {
        if (is_zone_edge(x, z, sz_location.x, sz_location.y, zone_dim))
        {
            // We are in one of the edges of the SZ
            hermite_check_conf conf{frame, params, class_costs, min_distances};
            heading = is_node ? NODE_HEADING(node_data, pos) : compute_heading(x, z, goal);
            cost = hermite_curve(frame_dim, {TO_FLOAT(x), TO_FLOAT(z), heading}, goal, wheelbase, delta_max_rad, &hermite_connection_check, &conf);
        }
        else
        {
            // nothing to do. We need to wait for the edge nodes to be processed
            if (is_node)
                SET_NODE_TYPE(node_conf, pos, NODE_TYPE_GRAPH_SZ_IN_CHECK);
            else
                SET_NODE_TYPE(node_conf, pos, NODE_TYPE_NULL_SZ_IN_CHECK);
            return;
        }
    }
    else
    {
        hermite_check_conf conf{frame, params, class_costs, min_distances};
        heading = is_node ? NODE_HEADING(node_data, pos) : compute_heading(x, z, goal);
        cost = hermite_curve(frame_dim, {TO_FLOAT(x), TO_FLOAT(z), heading}, goal, wheelbase, delta_max_rad, &hermite_connection_check, &conf);
    }

    if (cost < 0)
    {
        SET_NODE_COST_TO_GOAL(node_data, pos, -1);
        return;
    }

    SET_NODE_COST_TO_GOAL(node_data, pos, cost);

    // printf("valid cost @ pos= %d: %f\n", pos, cost);

    if (is_node)
        SET_NODE_TYPE(node_conf, pos, NODE_TYPE_GRAPH_CONNECTED_TO_GOAL);
    else
    {
        SET_NODE_HEADING(node_data, pos, heading);
        SET_NODE_TYPE(node_conf, pos, NODE_TYPE_NULL_CONNECTED_TO_GOAL);
    }

    // printf("valid cost @ pos= %d,%d: %f, heading: %f deg\n", x,z, cost, (heading * 180) / PI);
}

__device__ __host__ void to_goal_wave_2(float3 *frame,
                                        int *params,
                                        float *physical_params,
                                        float *class_costs,
                                        int pos,
                                        float3 goal,
                                        int4 *node_conf,
                                        float4 *node_data,
                                        uint4 *search_zone_info)
{

    const int t = NODE_TYPE(node_conf, pos);
    if (t != NODE_TYPE_GRAPH_SZ_IN_CHECK && t != NODE_TYPE_NULL_SZ_IN_CHECK)
        return;

    const int2 frame_dim = {params[FRAME_PARAM_WIDTH], params[FRAME_PARAM_HEIGHT]};
    const int2 zone_dim = {params[FRAME_SEARCH_ZONE_DIM_WIDTH], params[FRAME_SEARCH_ZONE_DIM_HEIGHT]};
    const int2 zone_grid_size = {params[FRAME_SEARCH_ZONE_GRID_WIDTH], params[FRAME_SEARCH_ZONE_GRID_HEIGHT]};
    const float wheelbase = physical_params[PHYSICAL_PARAM_WHEELBASE_PX];
    const float delta_max_rad = physical_params[PHYSICAL_PARAM_MAX_STEERING_RAD];

    int z = pos / frame_dim.x;
    int x = pos - z * frame_dim.x;

    int2 sz_location = zone_location(zone_dim, zone_grid_size, x, z);
    int4 edges_frame_pos = sz_egdes_frame_pos(sz_location, zone_dim, frame_dim.x);

    bool ignore_check = true;
    float total_cost = 0;
    float c = NODE_COST_TO_GOAL(node_data, edges_frame_pos.x);
    if (c < 0)
        ignore_check = false;
    total_cost += c;
    c = NODE_COST_TO_GOAL(node_data, edges_frame_pos.y);
    if (c < 0)
        ignore_check = false;
    total_cost += c;
    c = NODE_COST_TO_GOAL(node_data, edges_frame_pos.z);
    if (c < 0)
        ignore_check = false;
    total_cost += c;
    c = NODE_COST_TO_GOAL(node_data, edges_frame_pos.w);
    if (c < 0)
        ignore_check = false;
    total_cost += c;

    const bool is_node = IS_NODE(node_conf, pos);
    float heading = is_node ? NODE_HEADING(node_data, pos) : compute_heading(x, z, goal);

    if (ignore_check)
    {
        if (is_node)
            SET_NODE_TYPE(node_conf, pos, NODE_TYPE_GRAPH_CONNECTED_TO_GOAL);
        else
        {
            SET_NODE_HEADING(node_data, pos, heading);
            SET_NODE_TYPE(node_conf, pos, NODE_TYPE_NULL_CONNECTED_TO_GOAL);
        }
        SET_NODE_COST_TO_GOAL(node_data, pos, total_cost / 4);
        return;
    }

    const int2 min_distances = {params[FRAME_PARAM_MIN_DIST_X], params[FRAME_PARAM_MIN_DIST_Z]};
    hermite_check_conf conf{frame, params, class_costs, min_distances};
    heading = is_node ? NODE_HEADING(node_data, pos) : compute_heading(x, z, goal);
    float cost = hermite_curve(frame_dim, {TO_FLOAT(x), TO_FLOAT(z), heading}, goal, wheelbase, delta_max_rad, &hermite_connection_check, &conf);

    if (cost > 0)
    {
        if (is_node)
            SET_NODE_TYPE(node_conf, pos, NODE_TYPE_GRAPH_CONNECTED_TO_GOAL);
        else
        {
            SET_NODE_HEADING(node_data, pos, heading);
            SET_NODE_TYPE(node_conf, pos, NODE_TYPE_NULL_CONNECTED_TO_GOAL);
        }
        SET_NODE_COST_TO_GOAL(node_data, pos, cost);
    }
}
