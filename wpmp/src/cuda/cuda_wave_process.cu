#include "../wpmp_data.h"
#include "../../include/wpmp_graph.h"

extern __device__ __host__ void to_goal_wave(float3 *frame, int *params, float *class_costs, int pos, float3 goal, float wheelbase, float delta_max_rad, int4 *node_conf, float4 *node_data);

__global__ void to_goal_wave_cuda(
    float3 *frame,
    int *params,
    float *class_costs,
    float wheelbase,
    float delta_max_rad,
    int4 *node_conf,
    float4 *node_data,
    float3 goal)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = params[FRAME_PARAM_WIDTH];
    const int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    to_goal_wave(frame, params, class_costs, pos, goal, wheelbase, delta_max_rad, node_conf, node_data);
}

void WGraph::compute_goal_wave(
    float3 *frame,
    Waypoint &goal)
{

    int numBlocks = floor(_graph_size / THREADS_IN_BLOCK) + 1;

    float3 goalpoint = {
        TO_FLOAT(goal.x()),
        TO_FLOAT(goal.z()),
        TO_FLOAT(goal.heading().rad())};

    to_goal_wave_cuda<<<numBlocks, THREADS_IN_BLOCK>>>(frame, _search_space_params->get(), _class_costs->get(), //
                                                       _wheelbase, _max_steering_angle_rad,                   //
                                                       _node_conf->getPtr(), _node_data->getPtr(), goalpoint);

    CUDA(cudaDeviceSynchronize());
}