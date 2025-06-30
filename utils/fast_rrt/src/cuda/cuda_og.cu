#include "../../include/graph.h"
#include "../../include/cuda_params.h"
#include "../../include/math_utils.h"

#define USE_ROUGH_ANGLES 1

extern __device__ __host__ void setIntrinsicCostCuda(float3 *graphData, long pos, float cost);

__device__ __host__ inline int COMPUTE_POS(int width, int x, int z)
{
    return z * width + x;
}
__device__ __host__ inline bool CHECK_OUT_BOUNDARIES(int width, int height, int x, int z)
{
    return x < 0 || x >= width || z < 0 || z >= height;
}

__device__ __host__ inline bool checkValidPropagate(float3 *frame, float *classCosts, int width, int height, int x, int z)
{
    if (CHECK_OUT_BOUNDARIES(width, height, x, z))
        return false;

    const int pos = COMPUTE_POS(width, x, z);
    const int nodeClass = TO_INT(frame[pos].x);
    return classCosts[nodeClass] >= 0;
}
__device__ __host__ inline void setObstacle(float3 *frame, int width, int height, int x, int z)
{
    if (CHECK_OUT_BOUNDARIES(width, height, x, z))
        return;

    const int pos = COMPUTE_POS(width, x, z);
    frame[pos].z = 1.0f;
}
__device__ __host__ inline void propagateMinDistance(float3 *frame, float *classCosts, const int width, const int height, const int minDistance, int pos, int x, int z)
{
    const bool top = checkValidPropagate(frame, classCosts, width, height, x, z - 1);
    const bool bottom = checkValidPropagate(frame, classCosts, width, height, x, z + 1);
    const bool left = checkValidPropagate(frame, classCosts, width, height, x - 1, z);
    const bool right = checkValidPropagate(frame, classCosts, width, height, x + 1, z);
    const bool top_left = checkValidPropagate(frame, classCosts, width, height, x - 1, z - 1);
    const bool top_right = checkValidPropagate(frame, classCosts, width, height, x + 1, z - 1);
    const bool bottom_left = checkValidPropagate(frame, classCosts, width, height, x - 1, z + 1);
    const bool bottom_right = checkValidPropagate(frame, classCosts, width, height, x + 1, z + 1);

    frame[pos].z = 1.0f;

    if (!top && !bottom && !left && !right && !top_left && !top_right && !bottom_left && !bottom_right)
    {
        return;
    }

    for (int i = 1; i < minDistance; i++)
    {
        if (top)
        {
            setObstacle(frame, width, height, x, z - i);
        }
        if (bottom)
        {
            setObstacle(frame, width, height, x, z + i);
        }
        if (left)
        {
            setObstacle(frame, width, height, x - i, z);
        }
        if (right)
        {
            setObstacle(frame, width, height, x + i, z);
        }
        if (top_left)
        {
            setObstacle(frame, width, height, x - i, z - i);
        }
        if (top_right)
        {
            setObstacle(frame, width, height, x + i, z - i);
        }
        if (bottom_left)
        {
            setObstacle(frame, width, height, x - i, z + i);
        }
        if (bottom_right)
        {
            setObstacle(frame, width, height, x + i, z + i);
        }
    }
}

__global__ void __CUDA_compute_minimal_distance_boundaries(float3 *graphData, float3 *frame, float *classCosts, int *_searchSpaceParams, const int minDistance, bool copyIntrinsicCost)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int width = _searchSpaceParams[FRAME_PARAM_WIDTH];
    int height = _searchSpaceParams[FRAME_PARAM_HEIGHT];
    int lower_bound_ego_x = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_Z];

    if (pos >= width * height)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int nodeClass = TO_INT(frame[pos].x);

    if (copyIntrinsicCost)
    {
        setIntrinsicCostCuda(graphData, pos, frame[pos].y);
    }

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
    {
        return;
    }

    if (classCosts[nodeClass] < 0)
    {
        propagateMinDistance(frame, classCosts, width, height, minDistance, pos, x, z);
    }
}

void CudaGraph::computeBoundaries(float3 *og, bool copyIntrinsicCost)
{
#ifdef USE_ROUGH_ANGLES
    return;
#endif
    int size = _frame->width() * _frame->height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    int minDistanceX = _searchSpaceParams[FRAME_PARAM_MIN_DIST_X];
    int minDistanceZ = _searchSpaceParams[FRAME_PARAM_MIN_DIST_Z];

    const int minDistance = TO_INT(sqrtf(minDistanceX * minDistanceX + minDistanceZ * minDistanceZ));

    __CUDA_compute_minimal_distance_boundaries<<<numBlocks, THREADS_IN_BLOCK>>>(
        _frameData->getCudaPtr(),
        og,
        _classCosts,
        _searchSpaceParams,
        minDistance,
        copyIntrinsicCost);

    CUDA(cudaDeviceSynchronize());
}

__device__ __host__ bool checkFeasible(float3 *og, int width, int x, int z, float heading)
{
#ifdef USE_ROUGH_ANGLES
    int l = TO_INT(og[COMPUTE_POS(width, x, z)].z);
    
    // all angles are feasible
    if (l == 0xff) return true;

    int i = TO_INT(heading * PI_OVER_8_inv) + 3;
    float a = PI_OVER_8 * i;
    int left = -1;
    int right = -1;

    if (heading == a)
    {
        left = i;
    }
    else if (heading > a)
    {
        left = i;
        right = i + 1;
    }
    else
    {
        left = i - 1;
        right = i;
    }

    
    if (left >= 0)
    {
        if (!(l & (1 << left)))
            return false;
    }
    if (right >= 0)
    {
        if (!(l & (1 << right)))
            return false;
    }
    return true;

#else
    return og[COMPUTE_POS(width, x, z)].z == 0.0;
#endif
}
