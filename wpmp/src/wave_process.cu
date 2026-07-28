#include "wpmp_data.h"
#include <driveless/math_utils.h>

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

extern __device__ __host__ bool collision_check(float3 *frame, int *params, float *classCost, int2 min_distance, int x, int z, float angle_radians);

__device__ __host__ float hermite_connection_check(void *ptr, int x, int z, float heading)
{
    auto conf = (hermite_check_conf *)ptr;
    float3 *frame = conf->frame;
    int *params = conf->params;
    float *class_costs = conf->classCost;
    int2 min_distances = conf->min_distance;
    const int width = params[FRAME_PARAM_WIDTH];

    if (!collision_check(frame, params, class_costs, min_distances, x, z, heading))
        return -1;

    return frame[COMPUTE_POS(width, x, z)].y;
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

__device__ __host__ void to_goal_wave(float3 *frame, int *params, float *class_costs, int pos, float3 goal, float wheelbase, float delta_max_rad, int4 *node_conf, float4 *node_data)
{

    const int2 frame_dim = {params[FRAME_PARAM_WIDTH], params[FRAME_PARAM_HEIGHT]};
    int2 min_distances = {params[FRAME_PARAM_MIN_DIST_X], params[FRAME_PARAM_MIN_DIST_Z]};

    hermite_check_conf conf{frame, params, class_costs, min_distances};

    int z = pos / frame_dim.x;
    int x = pos - z * frame_dim.x;

    const bool is_node = IS_NODE(node_conf, pos);

    float heading = is_node ? NODE_HEADING(node_data, pos) : compute_heading(x, z, goal);

    float cost = hermite_curve(frame_dim, {TO_FLOAT(x), TO_FLOAT(z), heading}, goal, wheelbase, delta_max_rad, &hermite_connection_check, &conf);

    SET_NODE_COST_TO_GOAL(node_data, pos, cost);
    
    if (cost < 0) return;

    if (is_node)
        SET_NODE_TYPE(node_conf, pos, NODE_TYPE_GRAPH_CONNECTED_TO_GOAL);
    else {
        SET_NODE_HEADING(node_data, pos, heading);
        SET_NODE_TYPE(node_conf, pos, NODE_TYPE_NULL_CONNECTED_TO_GOAL);
    }

}
