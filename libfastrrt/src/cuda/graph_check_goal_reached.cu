
#include <driveless/cuda_basic.h>
#include <driveless/cuda_params.h>
#include <driveless/cuda_ptr.h>
#include "../../include/graph.h"

extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern __device__ __host__ float getCostCuda(float4 *graphData, long pos);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getHeadingCuda(float4 *graphData, long pos);
extern __device__ __host__ bool is_directly_connected_to_goal(float3 *goalDirectConnectionData, int width, int x, int z);
extern __device__ __host__ float checkDirectConnectionToGoal(float4 *graphData, float3 *frame, float *classCosts, int *searchSpaceParams, float max_curvature, int x, int z, float local_heading, int goal_x, int goal_z, float goal_heading, bool isSafeZoneChecked, bool isDistanceToGoalProcessed);
extern __device__ __host__ float get_heading_direct_connection_to_goal(float3 *goalDirectConnectionData, int width, int x, int z);
extern __device__ __host__ float get_cost_direct_connection_to_goal(float3 *goalDirectConnectionData, int width, int x, int z);
extern __device__ __host__ void setTypeCuda(int4 *graph, long pos, int type);

__global__ void __CUDA__check_goal_reached_with_direct_connection_cost(
    int4 *graph,
    float4 *graphData,
    float3 *frame,
    float3 *directConnection,
    int *params,
    float *classCost,
    float searchRadius,
    float max_curvature,
    bool safeZoneChecked,
    long long *bestCost)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].z != GRAPH_TYPE_NODE) // w means that the point is part of the graph
        return;

    float heading = getHeadingCuda(graphData, pos);

    for (int zp = z - searchRadius; zp < z + searchRadius; zp++)
        for (int xp = x - searchRadius; xp < x + searchRadius; xp++)
        {
            if (zp < 0 || zp >= height)
                continue;
            if (xp < 0 || xp >= width)
                continue;

            if (!is_directly_connected_to_goal(directConnection, width, xp, zp))
                continue;

            float local_intermediate_heading = get_heading_direct_connection_to_goal(directConnection, width, xp, zp);

            float cost = checkDirectConnectionToGoal(graphData, frame, classCost, params, max_curvature, x, z, heading, xp, zp, local_intermediate_heading, safeZoneChecked, false);

            if (cost < 0)
                continue;

            cost += get_cost_direct_connection_to_goal(directConnection, width, xp, zp);

            long long lcost = __float2ll_rd(100 * cost);
            atomicMin(bestCost, cost);
            setTypeCuda(graph, pos, GRAPH_TYPE_PROCESSING);
        }
}

__global__ void __CUDA__check_goal_reached_with_direct_connection(
    int4 *graph,
    float4 *graphData,
    float3 *frame,
    float3 *directConnection,
    int *params,
    float *classCost,
    float searchRadius,
    float max_curvature,
    bool safeZoneChecked,
    long long bestCost,
    int2 *node)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].z != GRAPH_TYPE_PROCESSING) // w means that the point is part of the graph
        return;

    setTypeCuda(graph, pos, GRAPH_TYPE_NODE);

    float heading = getHeadingCuda(graphData, pos);

    for (int zp = z - searchRadius; zp < z + searchRadius; zp++)
        for (int xp = x - searchRadius; xp < x + searchRadius; xp++)
        {
            if (zp < 0 || zp >= height)
                continue;
            if (xp < 0 || xp >= width)
                continue;

            float local_intermediate_heading = get_heading_direct_connection_to_goal(directConnection, width, xp, zp);

            float cost = checkDirectConnectionToGoal(graphData, frame, classCost, params, max_curvature, x, z, heading, xp, zp, local_intermediate_heading, safeZoneChecked, false);

            if (cost < 0)
                continue;

            cost += get_cost_direct_connection_to_goal(directConnection, width, xp, zp);

            long long lcost = __float2ll_rd(100 * cost);
            if (lcost <= bestCost)
            {
                node->x = x;
                node->y = z;
            }
        }
}

int2 CudaGraph::findBestGoalDirectConnection(float3 *og, angle heading, float radius, bool isSafeZoneChecked)
{
    int size = _graph->width() * _graph->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    CudaPtr<int2> bestNode(1);
    CudaPtr<long long> cost(1);

    bestNode.get()->x = -1;
    bestNode.get()->y = -1;
    *cost.get() = 99999999999;

    float max_curvature = _physicalParams->get()[PHYSICAL_MAX_CURVATURE];

    __CUDA__check_goal_reached_with_direct_connection_cost<<<numBlocks, THREADS_IN_BLOCK>>>(
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        og,
        _graphGoalDirectConnection->getCudaPtr(),
        _searchSpaceParams->get(),
        _classCosts->get(),
        radius,
        max_curvature,
        isSafeZoneChecked,
        cost.get());

    CUDA(cudaDeviceSynchronize());

    if (*cost.get() >= 99999999999)
        return {-1, -1};

    __CUDA__check_goal_reached_with_direct_connection<<<numBlocks, THREADS_IN_BLOCK>>>(
        _graph->getCudaPtr(),
        _graphData->getCudaPtr(),
        og,
        _graphGoalDirectConnection->getCudaPtr(),
        _searchSpaceParams->get(),
        _classCosts->get(),
        radius,
        max_curvature,
        isSafeZoneChecked,
        *cost.get(),
        bestNode.get());

    CUDA(cudaDeviceSynchronize());

    return {bestNode.get()->x, bestNode.get()->y};
}

