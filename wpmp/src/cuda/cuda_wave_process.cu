#include "../wpmp_data.h"
#include "../../include/wpmp_graph.h"

extern __device__ __host__ void to_goal_wave(float3 *frame,
                                             int *params,
                                             float *physical_params,
                                             float *class_costs,
                                             int pos,
                                             float3 goal,
                                             int4 *node_conf,
                                             float4 *node_data,
                                             uint4 *search_zone_info);

extern __device__ __host__ void to_goal_wave_2(float3 *frame,
                                               int *params,
                                               float *physical_params,
                                               float *class_costs,
                                               int pos,
                                               float3 goal,
                                               int4 *node_conf,
                                               float4 *node_data,
                                               uint4 *search_zone_info);

__global__ void to_goal_wave_cuda(
    float3 *frame,
    int *params,
    float *physical_params,
    float *class_costs,
    float3 goal,
    int4 *node_conf,
    float4 *node_data,
    uint4 *search_zone_info)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = params[FRAME_PARAM_WIDTH];
    const int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    to_goal_wave(frame, params, physical_params, class_costs, pos, goal, node_conf, node_data, search_zone_info);
}

__global__ void to_goal_wave_cuda_step2(
    float3 *frame,
    int *params,
    float *physical_params,
    float *class_costs,
    float3 goal,
    int4 *node_conf,
    float4 *node_data,
    uint4 *search_zone_info)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = params[FRAME_PARAM_WIDTH];
    const int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    to_goal_wave_2(frame, params, physical_params, class_costs, pos, goal, node_conf, node_data, search_zone_info);
}

void WGraph::compute_goal_wave(
    SearchFrame *frame,
    Waypoint &goal)
{

    int numBlocks = floor(_graph_size / THREADS_IN_BLOCK) + 1;

    int *_search_space_params = frame->getFrameParamsPtr();
    float *_class_costs = frame->getClassCostsPtr();
    uint4 *search_zone_info = frame->getSearchZonePtr();

    float3 goalpoint = {
        TO_FLOAT(goal.x()),
        TO_FLOAT(goal.z()),
        TO_FLOAT(goal.heading().rad())};

    to_goal_wave_cuda<<<numBlocks, THREADS_IN_BLOCK>>>(frame->getPtr(),
                                                       _search_space_params, frame->getPhysicalParamsPtr(), //
                                                       _class_costs, goalpoint, _node_conf->getPtr(),
                                                       _node_data->getPtr(), search_zone_info);

    CUDA(cudaDeviceSynchronize());

    to_goal_wave_cuda_step2<<<numBlocks, THREADS_IN_BLOCK>>>(frame->getPtr(),
                                                             _search_space_params, frame->getPhysicalParamsPtr(), //
                                                             _class_costs, goalpoint, _node_conf->getPtr(),
                                                             _node_data->getPtr(), search_zone_info);

    CUDA(cudaDeviceSynchronize());
}