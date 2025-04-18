#include "../../../cudac/include/cuda_basic.h"
#include "../../include/cuda_params.h"
#include "../../include/graph.h"

extern __device__ __host__ int2 getParentCuda(int4 *graph, long pos);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ void setTypeCuda(int4 *graph, long pos, int type);
extern __device__ __host__ int getTypeCuda(int4 *graph, long pos);

__global__ void __CUDA_solveGraphCollision_erase_trees(int4 *graph, int *params, int numNodesInGraph)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    if (getTypeCuda(graph, pos) != GRAPH_TYPE_NODE)
        return;

    for (int i = 0; i <= numNodesInGraph; i++)
    {
        int2 parent = getParentCuda(graph, pos);

        if (parent.x == -1 && parent.y == -1)
            return;

        long pos_parent = computePos(width, parent.x, parent.y);

        if (getTypeCuda(graph, pos_parent) == GRAPH_TYPE_COLLISION)
        {
            setTypeCuda(graph, pos, GRAPH_TYPE_NULL);
            return;
        }

        pos = pos_parent;
    }
}

__global__ void __CUDA_solveGraphCollision_set_nodes(int4 *graph, int *params)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    if (getTypeCuda(graph, pos) == GRAPH_TYPE_COLLISION)
        setTypeCuda(graph, pos, GRAPH_TYPE_NODE);
}

void CudaGraph::solveCollisions()
{
    return;
    int numNodesInGraph = count();

    if (numNodesInGraph <= 2)
        return;

    int size = _frame->width() * _frame->height();

    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_solveGraphCollision_erase_trees<<<numBlocks, THREADS_IN_BLOCK>>>(_frame->getCudaPtr(), _searchSpaceParams, numNodesInGraph);

    cudaDeviceSynchronize();

    __CUDA_solveGraphCollision_set_nodes<<<numBlocks, THREADS_IN_BLOCK>>>(_frame->getCudaPtr(), _searchSpaceParams);

    cudaDeviceSynchronize();
}