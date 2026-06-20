#include "../../include/cuda_basic.h"
#include "../../include/cpu_parallel_processor.h"
#include "../../include/search_frame_cpu.h"

#define NUM_POINTS_ON_MEAN 5

extern float ___computeMeanHeading(float4 *waypoints, int pos, int size, bool *valid, int width, int height);
extern float __computeHeading(int p1_x, int p1_y, int p2_x, int p2_y, bool *valid, int width, int height);
extern bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);
extern bool checkStateFeasible(float3 *searchFrame, int *params, float *classCosts, float4 *waypoints, int waypoints_size, int current_pos_waypoints, int minDistX, int minDistZ);
extern float __computeHeading(int p1_x, int p1_y, int p2_x, int p2_y, bool *valid, int width, int height);
extern float ___computeMeanHeading(float4 *waypoints, int pos, int size, bool *valid, int width, int height);
extern std::unique_ptr<float4[]> copyToCpuMemory(std::vector<Waypoint> points);
extern std::unique_ptr<float4[]> copyToCpuMemory(float *path, int count);

extern bool checkFeasiblePathCPU(float *points, int count, float3 *searchFrame, int *params, float *classCosts, int minDistX, int minDistZ);

class ParallelCheckFeasiblePath : public ParallelProcessor
{
    bool _pathFeasible;
    SearchFrameCPU *_searchFrame;
    float4 *_path;
    int _count;
    int *_params;
    float *_classCosts;
    int _minDistX;
    int _minDistZ;
    bool _informWaypointIndividualFeasibility;

public:
    ParallelCheckFeasiblePath(
        float4 *path, int pathSize,
        SearchFrameCPU *searchFrame,
        int *params, float *classCosts,
        int numThreadHandlers,
        int minDistX, int minDistZ)
        : ParallelProcessor(numThreadHandlers, 1, pathSize)
    {
        _pathFeasible = true;
        _searchFrame = searchFrame;
        _params = params;
        _classCosts = classCosts;
        _minDistX = minDistX;
        _minDistZ = minDistZ;
        _path = path;
        _count = pathSize;
    }

    void handler(int threadId) override
    {
        if (threadId >= _count)
            return;

        if (!checkStateFeasible(_searchFrame->getPtr(), _params, _classCosts, _path, _count, threadId, _minDistX, _minDistZ))
        {
            // printf("[CPU] unfeasible in %d, %d\n", x, z);
            _pathFeasible = false;
        }
    }

    bool isPathFeasible()
    {
        return _pathFeasible;
    }
};

bool SearchFrameCPU::checkFeasiblePath(std::vector<Waypoint> &path, int minDistX, int minDistZ, bool informWaypointIndividualFeasibility)
{
    std::unique_ptr<float4[]> ptr = copyToCpuMemory(path);

    auto parallel_checker = new ParallelCheckFeasiblePath(ptr.get(), path.size(), this, _params.get(), _classCosts.get(), _numCPUThreadHandlers, minDistX, minDistZ);
    parallel_checker->runAndWait();

    bool res = parallel_checker->isPathFeasible();

    if (informWaypointIndividualFeasibility)
    {
        if (res)
        {
            for (int i = 0; i < path.size(); i++)
            {
                path[i].set_checked_as_feasible(true);
            }
        }
        else
        {
            for (int i = 0; i < path.size(); i++)
            {
                path[i].set_checked_as_feasible(ptr.get()[i].w == 1.0);
            }
        }
    }
    ptr.reset();
    return res;
}

bool SearchFrameCPU::checkFeasiblePath(float *path, int count, int minDistX, int minDistZ, bool informWaypointIndividualFeasibility)
{
    std::unique_ptr<float4[]> ptr = copyToCpuMemory(path, count);

    auto parallel_checker = new ParallelCheckFeasiblePath(ptr.get(), count, this, _params.get(), _classCosts.get(), _numCPUThreadHandlers, minDistX, minDistZ);
    parallel_checker->runAndWait();

    bool res = parallel_checker->isPathFeasible();

    if (informWaypointIndividualFeasibility)
    {
        if (res)
        {
            for (int i = 0; i < count; i++)
            {
                int pos = 4 * i + 3;
                path[pos] = 1.0;
            }
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                int pos = 4 * i + 3;
                path[pos] = ptr.get()[i].w;
            }
        }
    }
    ptr.reset();
    return res;
}

bool SearchFrameCPU::computePathHeadings(int width, int height, std::vector<Waypoint> &waypoints)
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

