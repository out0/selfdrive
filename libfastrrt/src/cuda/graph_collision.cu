
#include <driveless/cuda_basic.h>
#include <driveless/cuda_params.h>
#include "../../include/graph.h"

extern __device__ __host__ int2 getParentCuda(int4 *graph, long pos);
extern __device__ __host__ long computePos(int width, int x, int z);
extern __device__ __host__ void setTypeCuda(int4 *graph, long pos, int type);
extern __device__ __host__ int getTypeCuda(int4 *graph, long pos);
extern __device__ __host__ void incNodeDeriveCount(int4 *graph, long pos);
extern __device__ __host__ void decNodeDeriveCount(int4 *graph, long pos);
extern __device__ __host__ void setNodeDeriveCount(int4 *graph, long pos, int count);

__global__ void __CUDA_solveGraphCollision_erase_trees(int4 *graph, int *params, int numNodesInGraph)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    if (getTypeCuda(graph, pos) != GRAPH_TYPE_NODE)
        return;

    int z = pos / width;
    int x = pos - z * width;
    //printf("[collision] solving for %d, %d\n", x, z);

    int curr = pos;
    for (int i = 0; i <= numNodesInGraph; i++)
    {
        int2 parent = getParentCuda(graph, curr);

        if (parent.x == -1 && parent.y == -1)
            return;

        long next = computePos(width, parent.x, parent.y);

        if (getTypeCuda(graph, next) == GRAPH_TYPE_COLLISION)
        {
            //printf("[collision] found collision parent of %d, %d in node %d, %d\n", x, z, parent.x, parent.y);
            setTypeCuda(graph, pos, GRAPH_TYPE_NULL);
            return;
        }

        curr = next;
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
    {
        setTypeCuda(graph, pos, GRAPH_TYPE_NODE);
        setNodeDeriveCount(graph, pos, 0);

        int2 parent = getParentCuda(graph, pos);
        long pos_parent = computePos(width, parent.x, parent.y);
        incNodeDeriveCount(graph, pos_parent);
    }
}

void CudaGraph::solveCollisions()
{
    int numNodesInGraph = count();

    if (numNodesInGraph <= 2)
        return;

    int size = _frame->width() * _frame->height();

    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_solveGraphCollision_erase_trees<<<numBlocks, THREADS_IN_BLOCK>>>(_frame->getCudaPtr(), _searchSpaceParams->get(), numNodesInGraph);

    cudaDeviceSynchronize();

    __CUDA_solveGraphCollision_set_nodes<<<numBlocks, THREADS_IN_BLOCK>>>(_frame->getCudaPtr(), _searchSpaceParams->get());

    cudaDeviceSynchronize();
}