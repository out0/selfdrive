#include "../../../cudac/include/cuda_basic.h"
#include "../../include/cuda_params.h"
#include "../../include/graph.h"

extern __device__ __host__ float4 draw_kinematic_path_candidate(int4 *graph, float3 *graphData, double *physicalParams, float3 *frame, float *classCosts, int width, int height, int2 center, int2 start, float steeringAngle, float pathSize, float velocity_m_s);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getHeadingCuda(float3 *graphData, long pos);
extern __device__ __host__ void setTypeCuda(int4 *graph, long pos, int type);
extern __device__ __host__ int getTypeCuda(int4 *graph, long pos);
extern __device__ __host__ int2 getParentCuda(int4 *graph, long pos);
extern __device__ __host__ void setCostCuda(float3 *graphData, long pos, float cost);
extern __device__ __host__ float getCostCuda(float3 *graphData, long pos);
extern __device__ __host__ bool set(int4 *graph, float3 *graphData, long pos, float heading, int parent_x, int parent_z, float cost, int type, bool override);
extern __device__ __host__ bool checkInGraphCuda(int4 *graph, long pos);
extern __device__ float generateRandom(curandState *state, int pos, float min_val, float max_val);
extern __device__ float generateRandomNeg(curandState *state, int pos, float max_val);
extern __device__ __host__ void setParentCuda(int4 *graph, long pos, int parent_x, int parent_z);
extern __device__ __host__ void incNodeDeriveCount(int4 *graph, long pos);
extern __device__ __host__ void setNodeDeriveCount(int4 *graph, long pos, int count);
extern __device__ __host__ int getNodeDeriveCount(int4 *graph, long pos);

extern __device__ __host__ float computeCost(float3 *frame, int4 *graph, float3 *graphData, double *physicalParams, float *classCosts, int width, float goalHeading_rad, long nodePos, double distToParent);

__device__ __host__ inline bool checkEquals(int2 &a, int2 &b)
{
    return a.x == b.x && a.y == b.y;
}


__global__ void __CUDA_KERNEL_acceptDerivatedPaths(int4 *graph, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    if (getTypeCuda(graph, pos) == GRAPH_TYPE_TEMP)
    {
        setTypeCuda(graph, pos, GRAPH_TYPE_NODE);
    }

    // atomicCAS(&(graph[pos].z), GRAPH_TYPE_TEMP, GRAPH_TYPE_NODE);
}

__global__ void __CUDA_KERNEL_randomlyDerivateNodes(curandState *state, int4 *graph, float3 *graphData, float3 *frame, float *classCosts, int width, int height, double *physicalParams, int2 gridCenter, float maxPathSize, float velocity_m_s, bool frontierExploration, bool *nodeCollision)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    if (!checkInGraphCuda(graph, pos))
        return;

    if (frontierExploration && getNodeDeriveCount(graph, pos) > 0)
    {
        // printf("%d, %d has been derived too many times, skipping...\n", x, z);
        return;
    }

    int z = pos / width;
    int x = pos - z * width;

    float heading = getHeadingCuda(graphData, pos);
    double maxSteeringAngle = physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];

    double steeringAngle = generateRandomNeg(state, pos, maxSteeringAngle);
    double pathSize = 0;

    while (pathSize <= 0)
    {
        pathSize = generateRandom(state, pos, 5.0, maxPathSize);
    }

    // TODO: support reverse by using a random variable and a flag to add a 180 degree turn on current heading before generating the kinematic path
    //       the problem with reverse is that we need an extra information (flag?) that tells that the movement is reverse in the graph.

    int2 start = {x, z};
    float4 end = draw_kinematic_path_candidate(graph, graphData, physicalParams, frame, classCosts, width, height, gridCenter, start, steeringAngle, pathSize, velocity_m_s);

    if (end.x < 0 || end.y < 0)
        return;

    int end_x = TO_INT(end.x);
    int end_z = TO_INT(end.y);
    float end_cost = end.z;
    float end_heading = end.w;

    long end_pos = computePos(width, end_x, end_z);

    if (end_pos == pos) return;


    if (checkInGraphCuda(graph, end_pos))
    {
        // printf ("[derive collision] %d, %d, %f + %f -> %d, %d, %f (rad: %f) size: %f\n", x, z, startHeading, steeringAngle, end.x, end.y, endHeading, getHeadingCuda(graphData, computePos(width, end.x, end.y)), pathSize);
        set(graph, graphData, end_pos, end_heading, x, z, end_cost, GRAPH_TYPE_COLLISION, true);
        setNodeDeriveCount(graph, pos, 1);
        *nodeCollision = true;
    }
    else
    {
        incNodeDeriveCount(graph, pos);
        set(graph, graphData, end_pos, end_heading, x, z, end_cost, GRAPH_TYPE_TEMP, true);
    }
}


void CudaGraph::acceptDerivedNodes()
{
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_acceptDerivatedPaths<<<numBlocks, THREADS_IN_BLOCK>>>(
        _frame->getCudaPtr(),
        _frame->width(),
        _frame->height());

    CUDA(cudaDeviceSynchronize());
}

void CudaGraph::acceptDerivedNode(int2 start, int2 lastNode)
{
    long pos = computePos(_frame->width(), lastNode.x, lastNode.y);
    setTypeCuda(_frame->getCudaPtr(), pos, GRAPH_TYPE_NODE);
}

void CudaGraph::expandTree(float3 *og, angle goalHeading, float maxPathSize, float velocity_m_s, bool frontierExpansion)
{
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    *_nodeCollision = false;

    __CUDA_KERNEL_randomlyDerivateNodes<<<numBlocks, THREADS_IN_BLOCK>>>(
        _randState,
        _frame->getCudaPtr(),
        _frameData->getCudaPtr(),
        og,
        _classCosts,
        _frame->width(),
        _frame->height(),
        _physicalParams,
        _gridCenter,
        maxPathSize,
        velocity_m_s,
        frontierExpansion, 
        _nodeCollision);

    CUDA(cudaDeviceSynchronize());

    if (*_nodeCollision)
    {
        //printf("Collision detected, solving...\n");
        solveCollisions();
    }
}

int2 CudaGraph::derivateNode(float3 *og, angle goalHeading, angle steeringAngle, double pathSize, float velocity_m_s, int x, int z)
{
    if (!checkInGraph(x, z))
        return int2{-1, -1};

    float4 p = draw_kinematic_path_candidate(_frame->getCudaPtr(), _frameData->getCudaPtr(), _physicalParams, og, _classCosts, _frame->width(), _frame->height(), _gridCenter, {x, z}, steeringAngle.rad(), pathSize, velocity_m_s);

    if (p.x < 0 || p.y < 0)
        return int2{-1, -1};
    

    int end_x = static_cast<int>(p.x);
    int end_z = static_cast<int>(p.y);
    float end_cost = p.z;
    float end_heading = p.w;


    long pos = computePos(width(), x, z);
    long pos_end = computePos(width(), end_x, end_z);

    if (checkInGraphCuda(_frame->getCudaPtr(), pos))
    {
        return {-1, -1};
        // printf ("[derive collision] %d, %d, %f + %f -> %d, %d, %f (rad: %f) size: %f\n", x, z, startHeading, steeringAngle, end.x, end.y, endHeading, getHeadingCuda(graphData, computePos(width, end.x, end.y)), pathSize);
        set(_frame->getCudaPtr(), _frameData->getCudaPtr(), pos_end, end_heading, x, z, end_cost, GRAPH_TYPE_COLLISION, true);
        setNodeDeriveCount(_frame->getCudaPtr(), pos, 1);
    }
    else
    {
        set(_frame->getCudaPtr(), _frameData->getCudaPtr(), pos_end, end_heading, x, z, end_cost, GRAPH_TYPE_TEMP, true);
        incNodeDeriveCount(_frame->getCudaPtr(), pos);
    }

    return {end_x, end_z};
}
