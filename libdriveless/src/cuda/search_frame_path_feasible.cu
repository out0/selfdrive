#include "../../include/search_frame.h"
#include "../../include/cuda_basic.h"



extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern __device__ __host__ bool checkStateFeasible(float3 *searchFrame, int *params, float *classCosts, float4 *waypoints, int waypoints_size, int current_pos_waypoints, int minDistX, int minDistZ);
extern __device__ __host__ float ___computeMeanHeading(float4 *waypoints, int pos, int size, bool *valid, int width, int height);


__global__ static void __CUDA_checkFeasiblePathGPU(float3 *searchFrame, int *params, float *classCosts, float4 *pathList, int size, bool *feasible, int minDistX, int minDistZ)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= size)
        return;

    if (!checkStateFeasible(searchFrame, params, classCosts, pathList, size, pos, minDistX, minDistZ))
        *feasible = false;
}

bool checkFeasiblePathCPU(float *points, int count, float3 *searchFrame, int *params, float *classCosts, int minDistX, int minDistZ)
{
    std::unique_ptr<float4[]> ptr = copyToCpuMemory(points, count);
    float4 *pathList = ptr.get();
    for (int i = 0; i < count; i++)
    {
        // int x = pathList[i].x;
        // int z = pathList[i].y;
        int pos = 4 * i + 3;
        points[pos] = 1.0;
        if (!checkStateFeasible(searchFrame, params, classCosts, pathList, count, i, minDistX, minDistZ))
        {
            points[pos] = 0.0;
            // printf("[CPU] unfeasible in %d, %d\n", x, z);
            return false;
        }
    }
    return true;
}
bool checkFeasiblePathGPU(float *points, int count, float3 *searchFrame, int *params, float *classCosts, int minDistX, int minDistZ, bool informWaypointIndividualFeasibility)
{
    cptr<float4> pathList = copyToCudaMemory(points, count);
    CudaPtr<bool> ptr(true);

    bool *f = ptr.get();
    *f = true;
    int numBlocks = floor(count / THREADS_IN_BLOCK) + 1;
    __CUDA_checkFeasiblePathGPU<<<numBlocks, THREADS_IN_BLOCK>>>(searchFrame, params, classCosts, pathList->get(), count, f, minDistX, minDistZ);
    CUDA(cudaDeviceSynchronize());

    if (informWaypointIndividualFeasibility)
        for (int i = 0; i < count; i++)
        {
            points[4 * i + 3] = pathList->get()[i].w;
        }

    return *f;
}

bool SearchFrame::checkFeasiblePath(std::vector<Waypoint> &path, int minDistX, int minDistZ, bool informWaypointIndividualFeasibility)
{
    int count = path.size();
    float *points = new float[count * 4];
    for (int i = 0; i < count; i++)
    {
        int pos = 4 * i;
        points[pos] = path[i].x();
        points[pos + 1] = path[i].z();
        points[pos + 2] = path[i].heading().rad();
        points[pos + 3] = 0.0;
    }
    bool res = checkFeasiblePath(points, count, minDistX, minDistZ, informWaypointIndividualFeasibility);

    if (informWaypointIndividualFeasibility)
        for (int i = 0; i < count; i++)
        {
            path[i].set_checked_as_feasible(points[4 * i + 3] == 1.0);
        }

    delete[] points;
    return res;
}

bool SearchFrame::checkFeasiblePath(float *points, int count, int minDistX, int minDistZ, bool informWaypointIndividualFeasibility)
{
    if (count == 0)
        return true;

    bool gpuExecution = count > PATH_FEASIBLE_CPU_THRESHOLD;

    int mx = TO_INT(minDistX / 2);
    int mz = TO_INT(minDistZ / 2);

    if (gpuExecution)
    {
        return checkFeasiblePathGPU(points, count, getPtr(), _params->get(), _classCosts->get(), mx, mz, informWaypointIndividualFeasibility);
    }
    return checkFeasiblePathCPU(points, count, getPtr(), _params->get(), _classCosts->get(), mx, mz);
}

bool SearchFrame::computePathHeadings(int width, int height, std::vector<Waypoint> &waypoints)
{
    std::unique_ptr<float4[]> ptr = copyToCpuMemory(waypoints);
    float4 *pathList = ptr.get();
    const int size = waypoints.size();
    for (int i = 0; i < size; i++)
    {
        bool valid = false;
        float heading = ___computeMeanHeading(ptr.get(), i, size, &valid, width, height);
        if (!valid)
            return false;
        waypoints[i].set_heading(heading + 0.0);
    }
    return true;
}
