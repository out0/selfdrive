#include "../wpmp_data.h"
#include "../../include/wpmp_graph.h"
#include <driveless/math_utils.h>
#include <driveless/search_zone_utils.h>

__device__ __host__ bool kinematic_curve(
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
    float max_heading_error_rad);

bool WGraph::expand(SearchFrame *frame, int x, int z, int max_size_px, Waypoint &goal, float max_error_dist_to_goal_px, float max_heading_error_rad)
{
    return false;
    // float node_cost = expand_node(
    //     frame->getPtr(),
    //     frame->getFrameParamsPtr(),
    //     frame->getPhysicalParamsPtr(),
    //     frame->getClassCostsPtr(),
    //     x, z,
    //     _node_conf->getPtr(),
    //     _node_data->getPtr(),
    //     max_size_px,
    //     {static_cast<float>(goal.x()), static_cast<float>(goal.z()), static_cast<float>(goal.heading().rad())},
    //     max_error_dist_to_goal_px,
    //     max_heading_error_rad,
    //     _best_cost.get());
}