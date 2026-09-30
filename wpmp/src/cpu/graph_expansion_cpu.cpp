#include "../wpmp_data.h"
#include "../../include/wpmp_graph.h"
#include <driveless/cuda_basic.h>
#include <driveless/math_utils.h>
#include <driveless/search_zone_utils.h>
#include "atomic_utils.h"

extern float expand_node(float3 *frame,
                         int *params,
                         float *physical_params,
                         float *class_costs,
                         int x, int z,
                         int4 *node_conf,
                         float4 *node_data,
                         int max_size_px,
                         float3 goal,
                         float max_dist_error_px,
                         float max_heading_error_rad);

bool WGraph::expand(SearchFrame *frame, int x, int z, int max_size_px, Waypoint &goal, float max_error_dist_to_goal_px, float max_heading_error_rad)
{
    float node_cost = expand_node(
        frame->getPtr(),
        frame->getFrameParamsPtr(),
        frame->getPhysicalParamsPtr(),
        frame->getClassCostsPtr(),
        x, z,
        _node_conf->getPtr(),
        _node_data->getPtr(),
        max_size_px,
        {static_cast<float>(goal.x()), static_cast<float>(goal.z()), static_cast<float>(goal.heading().rad())},
        max_error_dist_to_goal_px,
        max_heading_error_rad);

    if (node_cost > 0)
    {
#ifdef DRIVELESS_CUDA_ENABLED
        const long long p = *(_best_cost.get());
        const long long q = 100 * node_cost;

        if (p > q)
        {
            *(_best_cost.get()) = q;
        }
#else
        const long long p = _best_cost.load();
        const long long q = 100 * node_cost;
        atomicMin(_best_cost, q);
#endif
    }

    return node_cost > 0;
}