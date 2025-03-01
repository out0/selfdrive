
#include "../../include/graph.h"
#include "../../include/cuda_params.h"

extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ double getHeadingCuda(double3 *graphData, long pos);
extern __device__ __host__ int2 getParentCuda(int3 *graph, long pos);
extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int x, int z, float angle_radians);
extern __device__ __host__ double getCostCuda(double3 *graphData, long pos);
extern __device__ __host__ bool set(int3 *graph, double3 *graphData, long pos, double heading, int parent_x, int parent_z, double cost, int type, bool override);
extern __device__ __host__ bool checkKinematicPath(int3 *graph, double3 *graphData, float3 *frame, double *physicalParams, int *params, float *classCost, int2 center, int2 start, int2 end, float velocity_m_s, float maxSteeringAngle, double &final_heading);
extern __device__ __host__ double computeCost(float3 *frame, int3 *graph, double3 *graphData, double *physicalParams, float *classCosts, int width, float goalHeading_rad, long nodePos, double distToParent);

__device__ bool checkCyclicReference(int3 *graph, int width, long currentPos, long checkPos)
{
    long pos = currentPos;

    while (pos != checkPos)
    {
        int2 parent = getParentCuda(graph, pos);

        if (parent.x < 0 || parent.y < 0)
            return false;

        pos = computePos(width, parent.x, parent.y);
    }

    return true;
}
__global__ void __CUDA_KERNEL_optimizeGraphWithNode(int3 *graph, double3 *graphData, float3 *frame, double *physicalParams, int *params, float *classCost, int2 center, int2 parentCandidate, float radius, float velocity_m_s, float goalHeading_rad)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    float maxSteeringAngle = physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    if (graph[pos].z != GRAPH_TYPE_NODE) // w means that the point is part of the graph
        return;

    int dx = parentCandidate.x - x;
    int dz = parentCandidate.y - z;

    double dist = sqrtf(dx * dx + dz * dz);

    // printf("[%d, %d] check dist: %f > radius %f?\n", x, z, dist, radius);
    if (dist > radius)
        return;

    long parentCandidatePos = computePos(width, parentCandidate.y, parentCandidate.x);

    double parentCandidateHeading = getHeadingCuda(graphData, parentCandidatePos);

    // printf("[%d, %d] check_cyclic_reference\n", x, z);
    if (checkCyclicReference(graph, width, pos, parentCandidatePos))
    {
        // printf ("detected cyclic ref on connecting %d, %d to %d, %d\n", x, z, parentCandidate.x, parentCandidate.y);
        return;
    }

    double final_heading;

    //  printf("[%d, %d] check_kinematic_path\n", x, z);

    if (!checkKinematicPath(graph, graphData, frame, physicalParams, params, classCost, center, {x, z}, parentCandidate, velocity_m_s, maxSteeringAngle, final_heading))
        return;

    double current_cost = getCostCuda(graphData, pos);
    double new_cost = computeCost(frame, graph, graphData, physicalParams, classCost, width, goalHeading_rad, pos, dist);

    // double new_cost = getCostCuda(graphData, parentCandidatePos) + diff_cost;
    //  printf("%d, %d current_cost: %f, new cost: %f\n", x, z, current_cost, new_cost);

    if (current_cost <= new_cost)
    {
        // printf ("cost of connecting %d, %d to %d, %d is higher: %f vs %f\n", x, z, parentCandidate.x, parentCandidate.y, current_cost, new_cost);
        return;
    }

    //  printf("[%d, %d] will be optimized to new parent %d, %d\n", x, z, parent_candidate_x, parent_candidate_z);

    // we should optimize
    set(graph, graphData, pos, final_heading, parentCandidate.x, parentCandidate.y, new_cost, GRAPH_TYPE_NODE, true);
}

void CudaGraph::__optimizeGraph(float3 *og, int x, int z, float radius, float velocity_m_s, angle goalHeading)
{
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_optimizeGraphWithNode<<<numBlocks, THREADS_IN_BLOCK>>>(
        _frame->getCudaPtr(),
        _frameData->getCudaPtr(),
        og,
        _physicalParams,
        _searchSpaceParams,
        _classCosts,
        _gridCenter,
        {x, z},
        radius,
        velocity_m_s,
        goalHeading.rad());

    CUDA(cudaDeviceSynchronize());
}

/// @brief Optimizes the graph with the added new nodes, changing node parents for total cost reduction (RRT*)
/// @param searchFrame
/// @param radius
void CudaGraph::optimizeGraph(float3 *og, angle goalHeading, float radius, float velocity_m_s)
{
    std::pair<int2 *, int> res = __listNodes(GRAPH_TYPE_TEMP);

    int count = res.second;
    int2 *nodes = res.first;

    for (int i = 0; i < count; i++)
    {
        // printf("Optimizing %d, %d\n", nodes[i].x , nodes[i].y);

        __optimizeGraph(og, nodes[i].x, nodes[i].y, radius, velocity_m_s, goalHeading);
    }

    cudaFreeHost(nodes);
}