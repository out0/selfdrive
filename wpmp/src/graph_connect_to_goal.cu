#include "wpmp_data.h"
#include <driveless/math_utils.h>
#include <driveless/search_zone_utils.h>

__device__ __host__ bool check_connect_to_goal_cost(int *params, long pos, int4 *node_conf, float4 *node_data, float3 goal, float *)
{
    const int width = params[FRAME_PARAM_WIDTH];
    const int height = params[FRAME_PARAM_HEIGHT];

    if (NODE_TYPE(node_conf, pos) == NODE_TYPE_GRAPH_CONNECTED_TO_GOAL) {
        
    }
}