#include "../../include/search_frame.h"
#include "../../include/cuda_basic.h"

#define NUM_POINTS_ON_MEAN 5

// extern __global__  void __CUDA_KERNEL_checkFeasibleWaypoints(float3 *frame, int *params, float *classCost, float3 *waypoints, int count, bool computeHeadings);
__device__ __host__ float ___computeMeanHeading(float4 *waypoints, int pos, int size, bool *valid, int width, int height);
__device__ __host__ float __computeHeading(int p1_x, int p1_y, int p2_x, int p2_y, bool *valid, int width, int height);
__device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians)
{
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    float c = cosf(angle_radians);
    float s = sinf(angle_radians);

    // if (x == 108 && z == 46)
    // {
    //     printf("minDistX: %d, minDistZ: %d\n", minDistX, minDistZ);
    // }

    for (int i = -minDistZ; i <= minDistZ; i++)
        for (int j = -minDistX; j <= minDistX; j++)
        {
            int xl = TO_INT(j * c - i * s + x);
            int zl = TO_INT(j * s + i * c + z);

            if (xl < 0 || xl >= width)
                continue;

            if (zl < 0 || zl >= height)
                continue;

            if (xl >= lower_bound_ego_x && xl <= upper_bound_ego_x && zl >= upper_bound_ego_z && zl <= lower_bound_ego_z)
                continue;

            int segmentation_class = TO_INT(frame[zl * width + xl].x);

            if (classCost[segmentation_class] < 0)
            {
                // printf("(%d, %d) invalid on %d, %d segmentation_class: %d (x param = %f) class cost %f\n", x, z, xl, zl, segmentation_class, frame[zl * width + xl].x, classCost[segmentation_class]);
                return false;
            }
        }
    return true;
}

__device__ __host__ bool checkStateFeasible(float3 *searchFrame, int *params, float *classCosts, float4 *waypoints, int waypoints_size, int current_pos_waypoints, int minDistX, int minDistZ)
{
    float heading;

    const int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    const int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    const int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    const int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];
    // const int width = params[FRAME_PARAM_WIDTH];
    // const int height = params[FRAME_PARAM_HEIGHT];

    int x = waypoints[current_pos_waypoints].x;
    int z = waypoints[current_pos_waypoints].y;

    waypoints[current_pos_waypoints].w = 1.0;

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
    {
        return true;
    }

    // if (computeHeading)
    // {
    //     bool valid = true;
    //     heading = ___computeMeanHeading(waypoints, current_pos_waypoints, waypoints_size, &valid, width, height);
    //     waypoints[current_pos_waypoints].z = heading;
    // }
    // else
    heading = waypoints[current_pos_waypoints].z;
    // printf ("heading: %f\n", heading);

    bool res = __computeFeasibleForAngle(searchFrame, params, classCosts, minDistX, minDistZ, x, z, heading);

    if (!res)
        waypoints[current_pos_waypoints].w = 0.0;

    return res;
}

__global__ static void __CUDA_checkFeasiblePathGPU(float3 *searchFrame, int *params, float *classCosts, float4 *pathList, int size, bool *feasible, int minDistX, int minDistZ)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= size)
        return;

    if (!checkStateFeasible(searchFrame, params, classCosts, pathList, size, pos, minDistX, minDistZ))
        *feasible = false;
}

__host__ cptr<float4> copyToCudaMemory(float *path, int count)
{
    cptr<float4> data = std::make_unique<CudaPtr<float4>>(count);
    float4 *addr = data->get();

    for (int c = 0; c < count; c++)
    {
        int pos = 4 * c;
        addr[c].x = path[pos];
        addr[c].y = path[pos + 1];
        addr[c].z = path[pos + 2];
        addr[c].w = 0.0;
    }
    return data;
}

__host__ std::unique_ptr<float4[]> copyToCpuMemory(std::vector<Waypoint> points)
{
    int count = points.size();
    std::unique_ptr<float4[]> ptr = std::make_unique<float4[]>(count);
    float4 *addr = ptr.get();

    for (int c = 0; c < count; c++)
    {
        addr[c].x = points[c].x();
        addr[c].y = points[c].z();
        addr[c].z = points[c].heading().rad();
        addr[c].w = 0.0;
    }
    return ptr;
}

__host__ std::unique_ptr<float4[]> copyToCpuMemory(float *path, int count)
{
    std::unique_ptr<float4[]> ptr = std::make_unique<float4[]>(count);
    float4 *addr = ptr.get();

    for (int c = 0; c < count; c++)
    {
        int pos = 4 * c;
        addr[c].x = path[pos];
        addr[c].y = path[pos + 1];
        addr[c].z = path[pos + 2];
        addr[c].w = 0.0;
    }
    return ptr;
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
        return checkFeasiblePathGPU(points, count, getCudaPtr(), _params->get(), _classCosts->get(), mx, mz, informWaypointIndividualFeasibility);
    }
    return checkFeasiblePathCPU(points, count, getCudaPtr(), _params->get(), _classCosts->get(), mx, mz);
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

__device__ __host__ float __computeHeading(int p1_x, int p1_y, int p2_x, int p2_y, bool *valid, int width, int height)
{
    *valid = false;
    if (p1_x == p2_x && p1_y == p2_y)
        return 0.0;

    if (p1_x < 0 || p1_y < 0 || p2_x < 0 || p2_y < 0)
        return 0.0;

    if (p1_x >= width || p1_y >= height || p2_x >= width || p2_y >= height)
        return 0.0;

    float dx = p2_x - p1_x;
    float dz = p2_y - p1_y;
    *valid = true;
    float heading = CUDART_PI_F / 2 - atan2f(-dz, dx);

    if (heading > CUDART_PI_F) // greater than 180 deg
        heading = heading - 2 * CUDART_PI_F;

    return heading;
}

__device__ __host__ float ___computeMeanHeading(float4 *waypoints, int pos, int size, bool *valid, int width, int height)
{
    float heading = 0.0;
    int count = 0;

    for (int j = 1; j <= NUM_POINTS_ON_MEAN; j++)
    {
        bool v = false;

        if (pos + j >= size)
            break;

        heading += __computeHeading((int)waypoints[pos].x, (int)waypoints[pos].y, (int)waypoints[pos + j].x, (int)waypoints[pos + j].y, &v, width, height);

        if (!v)
            break;

        count++;
    }

    if (count != NUM_POINTS_ON_MEAN)
    {
        count = 0;
        // compute in reverse
        for (int j = 1; j <= NUM_POINTS_ON_MEAN; j++)
        {
            bool v = false;
            if (pos - j < 0)
            {
                *valid = false;
                return 0.0;
            }
            heading += __computeHeading((int)waypoints[pos - j].x, (int)waypoints[pos - j].y, (int)waypoints[pos].x, (int)waypoints[pos].y, &v, width, height);
            if (!v)
                break;
            count++;
        }
    }

    *valid = count > 0;

    if (*valid)
        return heading / count;

    return 0.0;
}
