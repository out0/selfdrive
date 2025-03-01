#include "../../../cudac/include/cuda_basic.h"
#include "../../include/cuda_params.h"
#include "../../include/graph.h"

extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int x, int z, float angle_radians);
extern __device__ __host__ double getCostCuda(double3 *graphData, long pos);

__global__ void __CUDA_KERNEL_findBestNodeWithHeading_bestCost(int3 *graph, double3 *graphData, float3 *frame, int *params, float *classCost, long long searchRadiusSq, int targetX, int targetZ, float targetHeading, long long *bestCost)
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

    int dx = targetX - x;
    int dz = targetZ - z;

    long long dist = __float2ll_rd(dx * dx + dz * dz);

    if (dist > searchRadiusSq)
        return;

    if (!__computeFeasibleForAngle(frame, params, classCost, x, z, targetHeading))
        return;

    // self cost + dist
    long long cost = __float2ll_rd(sqrtf(dist) + getCostCuda(graphData, pos));

    atomicMin(bestCost, cost);
}

__global__ void __CUDA_KERNEL_findBestNodeWithHeading_firstNodeWithCost(int3 *graph, double3 *graphData, float3 *frame, int *params, float *classCost, long long searchRadiusSq, int targetX, int targetZ, float targetHeading, long long bestCost, int2 *node)
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

    int dx = targetX - x;
    int dz = targetZ - z;

    long long dist = __float2ll_rd(dx * dx + dz * dz);

    if (dist > searchRadiusSq)
        return;

    if (!__computeFeasibleForAngle(frame, params, classCost, x, z, targetHeading))
        return;

    // self cost + dist
    long long cost = __float2ll_rd(sqrtf(dist) + getCostCuda(graphData, pos));

    if (cost == bestCost)
    {
        (*node).x = x;
        (*node).y = z;
    }
}

int2 CudaGraph::findBestNode(float3 *og, angle heading, float radius, int x, int z)
{
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    int2 *bestNode;
    long long *cost;

    if (!cudaAllocMapped(&bestNode, sizeof(int2)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(int2)) + std::string(" bytes for findBestNode\n");
        throw msg;
    }
    if (!cudaAllocMapped(&cost, sizeof(long long)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(long long)) + std::string(" bytes for findBestNode\n");
        throw msg;
    }

    bestNode->x = -1;
    bestNode->y = -1;
    *cost = 99999999999;

    __CUDA_KERNEL_findBestNodeWithHeading_bestCost<<<numBlocks, THREADS_IN_BLOCK>>>(
        _frame->getCudaPtr(),
        _frameData->getCudaPtr(),
        og,
        _searchSpaceParams,
        _classCosts,
        static_cast<long long>(radius * radius),
        x, z, heading.rad(), cost);

    CUDA(cudaDeviceSynchronize());

    if (*cost >= 99999999999)
    {
        cudaFreeHost(bestNode);
        cudaFreeHost(cost);
        return {-1, -1};
    }

    __CUDA_KERNEL_findBestNodeWithHeading_firstNodeWithCost<<<numBlocks, THREADS_IN_BLOCK>>>(
        _frame->getCudaPtr(),
        _frameData->getCudaPtr(),
        og,
        _searchSpaceParams,
        _classCosts,
        static_cast<long long>(radius * radius),
        x, z, heading.rad(), *cost, bestNode);

    CUDA(cudaDeviceSynchronize());

    int2 resp = {bestNode->x, bestNode->y};
    cudaFreeHost(bestNode);
    cudaFreeHost(cost);

    return resp;
}



extern __device__ __host__ double compute_euclidean_2d_dist(int2 &start, int2 &end);

__global__ void __CUDA_KERNEL_checkGoalReached(int3 *graph, float3 * frame, int *params, float *costs,  int goalX, int goalZ, float heading_rad, float distToGoalTolerance, bool *goalReached)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    if (graph[pos].z != GRAPH_TYPE_NODE)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int2 s = {x, z};
    int2 e = {goalX, goalZ};

    if (compute_euclidean_2d_dist(s, e) >  distToGoalTolerance)
        return;

    if (__computeFeasibleForAngle(frame, params, costs, x, z, heading_rad)) {
        *goalReached = true;
    }

    //printf ("%d, %d is not feasible for angle %f\n", x, z, 180 * heading_rad / PI);

    // atomicCAS(&(graph[pos].z), GRAPH_TYPE_TEMP, GRAPH_TYPE_NODE);
}

bool CudaGraph::checkGoalReached(float3 *og, int2 goal, angle heading, float distanceToGoalTolerance) {
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    if (*_goalReached) return true;
   
    __CUDA_KERNEL_checkGoalReached<<<numBlocks, THREADS_IN_BLOCK>>>(
        _frame->getCudaPtr(),
        og,
        _searchSpaceParams,
        _classCosts,
        goal.x,
        goal.y,
        heading.rad(),
        distanceToGoalTolerance,
        _goalReached);

    CUDA(cudaDeviceSynchronize());
    return *_goalReached;
}