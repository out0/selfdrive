#include "../../../cudac/include/cuda_basic.h"
#include "../../include/cuda_params.h"
#include "../../include/graph.h"

extern __device__ __host__ int2 draw_kinematic_path_candidate(int3 *graph, float3 *graphData, double *physicalParams, float3 *frame, float *classCosts, int width, int height, int2 center, int2 start, float steeringAngle, float pathSize, float velocity_m_s);
extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int x, int z, float angle_radians);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ double getHeadingCuda(float3 *graphData, long pos);
extern __device__ __host__ void setTypeCuda(int3 *graph, long pos, int type);
extern __device__ __host__ int getTypeCuda(int3 *graph, long pos);
extern __device__ __host__ int2 getParentCuda(int3 *graph, long pos);
extern __device__ __host__ void setCostCuda(float3 *graphData, long pos, double cost);
extern __device__ __host__ double getCostCuda(float3 *graphData, long pos);
extern __device__ __host__ bool set(int3 *graph, float3 *graphData, long pos, double heading, int parent_x, int parent_z, double cost, int type, bool override);
extern __device__ __host__ bool checkInGraphCuda(int3 *graph, long pos);
extern __device__ float generateRandom(curandState *state, int pos, float max);
extern __device__ float generateRandomNeg(curandState *state, int pos, float max);
extern __device__ __host__ void setParentCuda(int3 *graph, long pos, int parent_x, int parent_z);

extern __device__ __host__ double computeCost(float3 *frame, int3 *graph, float3 *graphData, double *physicalParams, float *classCosts, int width, float goalHeading_rad, long nodePos, double distToParent);

__device__ __host__ inline bool checkEquals(int2 &a, int2 &b)
{
    return a.x == b.x && a.y == b.y;
}

__device__ void parallel_check_path_node(int3 *graph, float3 *graphData, float3 *cudaFrame, int *params, float *classCost, int type, int x, int z)
{

    int width = params[FRAME_PARAM_WIDTH];
    long pos = computePos(width, x, z);

    double heading = getHeadingCuda(graphData, pos);
    int2 parent = getParentCuda(graph, pos);

    bool finalNode = type == GRAPH_TYPE_TEMP;
    bool feasible = __computeFeasibleForAngle(cudaFrame, params, classCost, x, z, heading);

    if (finalNode)
    {
        if (!feasible)
            setTypeCuda(graph, pos, GRAPH_TYPE_NULL);
        return;
    }

    if (!feasible)
    {
        long parentPos = computePos(width, parent.x, parent.y);
        setTypeCuda(graph, parentPos, GRAPH_TYPE_NULL);
    }

    setTypeCuda(graph, pos, GRAPH_TYPE_NULL);
}

__global__ void __CUDA_KERNEL_checkDerivatedPaths(int3 *graph, float3 *graphData, float3 *cudaFrame, int *params, float *classCost)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int type = getTypeCuda(graph, pos);

    if (type == GRAPH_TYPE_NULL || type == GRAPH_TYPE_NODE)
        return;

    parallel_check_path_node(graph, graphData, cudaFrame, params, classCost, type, x, z);
}

__global__ void __CUDA_KERNEL_acceptDerivatedPaths(int3 *graph, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    if (graph[pos].z == GRAPH_TYPE_TEMP)
    {
        graph[pos].z = GRAPH_TYPE_NODE;
    }

    // atomicCAS(&(graph[pos].z), GRAPH_TYPE_TEMP, GRAPH_TYPE_NODE);
}

__device__ void prepare_path_candidate_for_parallel_check(float3 *frame, int3 *graph, float3 *graphData, float *classCosts, double *physicalParams, int width, int height, int2 start, int2 end, float goalHeading_rad, float pathSize)
{
    if (checkEquals(start, end))
        return;
    long pos = computePos(width, end.x, end.y);
    int2 parent = getParentCuda(graph, pos);
    double heading = getHeadingCuda(graphData, pos);

    double nodeCost = computeCost(frame, graph, graphData, physicalParams, classCosts, width, goalHeading_rad, pos, pathSize);
    set(graph, graphData, pos, heading, start.x, start.y, pos, GRAPH_TYPE_TEMP, true);

    while (parent.x != start.x || parent.y != start.y)
    {
        pos = computePos(width, parent.x, parent.y);
        // copy the next parent to use in the next iteraction
        parent.x = graph[pos].x;
        parent.y = graph[pos].y;
        // updates the current parent to point to the last node.
        graph[pos].x = end.x;
        graph[pos].y = end.y;
    }
}

__global__ void __CUDA_KERNEL_randomlyDerivateNodes(curandState *state, int3 *graph, float3 *graphData, float3 *frame, float *classCosts, int width, int height, double *physicalParams, int2 gridCenter, float maxPathSize, float velocity_m_s, float goalHeading_rad)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    if (!checkInGraphCuda(graph, pos))
        return;


    int z = pos / width;
    int x = pos - z * width;

    double heading = getHeadingCuda(graphData, pos);
    double maxSteeringAngle = physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];

    double steeringAngle = generateRandomNeg(state, pos, maxSteeringAngle);
    double pathSize = 0;

    while (pathSize <= 0)
    {
        pathSize = generateRandom(state, pos, maxPathSize);
    }

    // TODO: support reverse by using a random variable and a flag to add a 180 degree turn on current heading before generating the kinematic path
    //       the problem with reverse is that we need an extra information (flag?) that tells that the movement is reverse in the graph.

    int2 start = {x, z}; 
    int2 end = draw_kinematic_path_candidate(graph, graphData, physicalParams, frame, classCosts, width, height, gridCenter, start, steeringAngle, pathSize, velocity_m_s);

    if (end.x < 0 || end.y < 0)
        return;

    prepare_path_candidate_for_parallel_check(frame, graph, graphData, classCosts, physicalParams, width, height, start, end, goalHeading_rad, pathSize);
}

void CudaGraph::__checkDerivatedPath(float3 *og)
{
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_checkDerivatedPaths<<<numBlocks, THREADS_IN_BLOCK>>>(
        _frame->getCudaPtr(),
        _frameData->getCudaPtr(),
        og,
        _searchSpaceParams,
        _classCosts);

    CUDA(cudaDeviceSynchronize());
}

void CudaGraph::acceptDerivatedNodes()
{
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_acceptDerivatedPaths<<<numBlocks, THREADS_IN_BLOCK>>>(
        _frame->getCudaPtr(),
        _frame->width(),
        _frame->height());

    CUDA(cudaDeviceSynchronize());
}

bool CudaGraph::__checkDerivatedPath(float3 *og, int2 start, int2 lastNode)
{
    int2 node;
    node.x = lastNode.x;
    node.y = lastNode.y;

    bool feasible = true;

    int width = _frame->width();

    if (checkEquals(start, lastNode))
        return false;

    while ((node.x != start.x || node.y != start.y) && node.x != -1 && node.y != -1)
    {
        long pos = computePos(width, node.x, node.y);
        double heading = getHeadingCuda(_frameData->getCudaPtr(), pos);
        feasible = feasible && __computeFeasibleForAngle(og, _searchSpaceParams, _classCosts, node.x, node.y, heading);
        setTypeCuda(_frame->getCudaPtr(), pos, GRAPH_TYPE_NULL);
        node = getParentCuda(_frame->getCudaPtr(), pos);
    }

    return feasible;
}

void CudaGraph::acceptDerivatedNode(int2 start, int2 lastNode)
{
    long pos = computePos(_frame->width(), lastNode.x, lastNode.y);
    setTypeCuda(_frame->getCudaPtr(), pos, GRAPH_TYPE_NODE);
}

void CudaGraph::derivateNode(float3 *og, angle goalHeading, float maxPathSize, float velocity_m_s)
{
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    // printf ("_randState: %p\n", (void *)_randState);
    // printf ("_frame_ptr: %p\n", (void *)_frame->getCudaPtr());
    // printf ("_frameData_ptr: %p\n", (void *)_frameData->getCudaPtr());
    // printf ("og (_ptr): %p\n", (void *)og);
    // printf ("_classCosts: %p\n", (void *)_classCosts);
    // printf ("_physicalParams: %p\n", (void *)_physicalParams);
    // printf ("w: %d, h:%d\n", _frame->width(), _frame->height());

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
        goalHeading.rad());

    CUDA(cudaDeviceSynchronize());

    __checkDerivatedPath(og);
}

int2 CudaGraph::derivateNode(float3 *og, angle goalHeading, angle steeringAngle, double pathSize, float velocity_m_s, int x, int z)
{
    if (!checkInGraph(x, z))
        return int2{-1, -1};

    int2 p = draw_kinematic_path_candidate(_frame->getCudaPtr(), _frameData->getCudaPtr(), _physicalParams, og, _classCosts, _frame->width(), _frame->height(), _gridCenter, {x, z}, steeringAngle.rad(), pathSize, velocity_m_s);

    if (__checkDerivatedPath(og, {x, z}, p))
    {
        long pos = computePos(_frame->width(), p.x, p.y);
        setTypeCuda(_frame->getCudaPtr(), pos, GRAPH_TYPE_TEMP);
        setParentCuda(_frame->getCudaPtr(), pos, x, z);

        double nodeCost = computeCost(og, _frame->getCudaPtr(), _frameData->getCudaPtr(), _physicalParams, _classCosts, width(), goalHeading.rad(), pos, pathSize);
        setCostCuda(_frameData->getCudaPtr(), pos, nodeCost);
        return p;
    }

    return {-1, -1};
}
