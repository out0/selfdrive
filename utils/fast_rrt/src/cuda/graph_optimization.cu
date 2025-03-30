
#include "../../include/graph.h"
#include "../../include/cuda_params.h"

void CudaGraph::optimizeGraph(float3 *og, angle goalHeading, float radius, float velocity_m_s)
{
    // Generate node candidates
    derivateNode(og, goalHeading, radius, velocity_m_s);

    int numNodesInGraph = count();

    // list all feasible node candidates
    std::pair<int2 *, int> lst = __listNodes(GRAPH_TYPE_TEMP);

    int count = lst.second;
    int2 * cudaResult = lst.first;

    numNodesInGraph += count; // add candidates

    for (int i = 0; i < count; i++)
    {
        optimizeNode(og, cudaResult[i].x, cudaResult[i].y, radius, velocity_m_s, numNodesInGraph);
    }

    cudaFreeHost(cudaResult);
    acceptDerivatedNodes();
}

extern __device__ __host__ int2 getParentCuda(int3 *graph, long pos);
extern __device__ __host__ void setParentCuda(int3 *graph, long pos, int parent_x, int parent_z);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ float getCostCuda(float3 *graphData, long pos);
extern __device__ __host__ float getHeadingCuda(float3 *graphData, long pos);
extern __device__ __host__ bool check_graph_connection(
    int3 *graph, 
    float3 *graphData, 
    float3 *frame, 
    double *physicalParams, 
    int *params, 
    float *classCost, 
    int2 center, 
    int2 start, 
    int2 end, 
    float velocity_m_s,
    float path_heading,
    float &path_cost);

__device__ __host__ bool checkCyclicRef(int3 *graph, int width, int height, int x, int z, int xc, int zc, int numNodesInGraph) {

    int xi = xc;
    int zi = zc;
    for (int i = 0; i <= numNodesInGraph; i++) {
        int2 parent = getParentCuda(graph, computePos(width, xi, zi));
        if (parent.x == x && parent.y == z) return true; // we've reached (x,z) from (xc, zc), so there's a cyclic ref.
        if (parent.x == -1 && parent.y == -1) return false;  // we've reached the origin
        xi = parent.x;
        zi = parent.y;
    }
    // the halt problem is avoided by returning true if we hopped for more than numNodesInGraph:
    return true;
}

__global__ static void __CUDA_KERNEL_optimize(
    int3 *graph, 
    float3* graphData, 
    float3 * frame, 
    double *physicalParams, 
    int *params,
    float *classCost,
    int2 center,
    int xc, 
    int zc, 
    float radius_sqr, 
    float velocity_m_s, 
    int numNodesInGraph)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (graph[pos].z != GRAPH_TYPE_NODE)
        return;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    // exit if node is parent
    int2 parent = getParentCuda(graph, pos);
    if (parent.x == xc && parent.y == zc) return;

    int z = pos / width;
    int x = pos - z * width;
    // exit if is the same node as the candidate
    if (x == xc && z == zc) return;

    // exit if not in range
    float dx = x - xc;
    float dz = z - zc;
    float dist_sq = dx * dx + dz * dz;
    if (dist_sq > radius_sqr)
        return;

    //check cyclic ref
    if (checkCyclicRef(graph, width, height, x, z, xc, zc, numNodesInGraph))
        return;
    
    float maxPathSize = 2 * sqrtf(dist_sq);

    //check if we can connect the node N in range with x,z as its parent, maintaining heading (x,z) as starting heading and heading_N as arriving reading
    // and having total size <= maxPathSize

    float newPathCost = 0.0; // TODO
    float heading = getHeadingCuda(graphData, pos);

    if (!check_graph_connection(graph, graphData, frame, 
        physicalParams, params, classCost, 
        center, 
        {xc, zc}, 
        {x, z}, 
        velocity_m_s,
        heading,
        newPathCost)) return;  

    float newCost = getCostCuda(graphData, computePos(width, xc, zc)) + newPathCost;
    float currentCost = getCostCuda(graphData, pos);
    
    if (newCost < currentCost) {
        //connect N to xc, zc
        setParentCuda(graph, pos, xc, zc);
    }
}


void CudaGraph::optimizeNode(float3 *og, int x, int z, float radius, float velocity_m_s, int numNodesInGraph)
{
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_optimize<<<numBlocks, THREADS_IN_BLOCK>>>(
        _frame->getCudaPtr(), 
        _frameData->getCudaPtr(),
        og,
        _physicalParams,
        _searchSpaceParams,
        _classCosts,
        _gridCenter,
        x,
        z,
        radius*radius,
        velocity_m_s,
        numNodesInGraph);

    CUDA(cudaDeviceSynchronize());
    
}
