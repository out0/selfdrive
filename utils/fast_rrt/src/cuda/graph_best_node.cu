#include "../../../cudac/include/cuda_basic.h"
#include "../../include/cuda_params.h"
#include "../../include/graph.h"

extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int x, int z, float angle_radians);
extern __device__ __host__ float getCostCuda(float3 *graphData, long pos);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getHeadingCuda(float3 *graphData, long pos);
extern __device__ __host__ bool checkFeasible(float3 *og, int width, int x, int z, float heading);

__global__ void __CUDA_KERNEL_findBestNodeWithHeading_bestCost(int4 *graph, float3 *graphData, float3 *frame, int *params, float *classCost, long long searchRadiusSq, int targetX, int targetZ, float targetHeading, long long *bestCost)
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
        
    
    float heading = getHeadingCuda(graphData, pos);

    if (!checkFeasible(frame, width, x, z, heading))
    {
        return;
    }

    if (abs(heading - targetHeading) > 0.035)
        return;

    // if (!__computeFeasibleForAngle(frame, params, classCost, x, z, targetHeading))
    //     return;

    // self cost + dist
    long long cost = __float2ll_rd(sqrtf(dist) + getCostCuda(graphData, pos));
    // printf("best node candidate: %d,%d: cost %f dist: %f, total cost: %ld\n",
    //     x,z,
    //     getCostCuda(graphData, pos),
    //     sqrtf(dist),
    //     cost
    // );

    atomicMin(bestCost, cost);
}

__global__ void __CUDA_KERNEL_findBestNodeWithHeading_firstNodeWithCost(int4 *graph, float3 *graphData, float3 *frame, int *params, float *classCost, long long searchRadiusSq, int targetX, int targetZ, float targetHeading, long long bestCost, int2 *node)
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

    // if (!__computeFeasibleForAngle(frame, params, classCost, x, z, targetHeading))
    //     return;
    float heading = getHeadingCuda(graphData, pos);
    if (!checkFeasible(frame, width, x, z, heading))
    {
        return;
    }

    if (abs(heading - targetHeading) > 0.035)
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

extern __device__ __host__ double compute_euclidean_2d_dist(const int2 &start, const int2 &end);

__global__ void __CUDA_KERNEL_checkGoalReached(int4 *graph, float3 *graphData, float3 *frame, int *params, float *costs, int goalX, int goalZ, float goalHeading, float distToGoalTolerance, bool *goalReached)
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

    if (compute_euclidean_2d_dist(s, e) > distToGoalTolerance)
        return;

    float heading = getHeadingCuda(graphData, pos);

    if (abs(heading - goalHeading) <= 0.035)
        *goalReached = true;
    // if (frame[computePos(width, x, z)].z == 0.0)
    // {
        
    // }

    // if (__computeFeasibleForAngle(frame, params, costs, x, z, heading_rad)) {
    //     *goalReached = true;
    // }

    // printf ("%d, %d is not feasible for angle %f\n", x, z, 180 * heading_rad / PI);

    // atomicCAS(&(graph[pos].z), GRAPH_TYPE_TEMP, GRAPH_TYPE_NODE);
}

bool CudaGraph::checkGoalReached(float3 *og, int2 goal, angle heading, float distanceToGoalTolerance)
{
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    if (*_goalReached)
        return true;

    // printf("check goal: %d, %d\n", goal.x, goal.y);

    __CUDA_KERNEL_checkGoalReached<<<numBlocks, THREADS_IN_BLOCK>>>(
        _frame->getCudaPtr(),
        _frameData->getCudaPtr(),
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