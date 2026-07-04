#include "../include/cuda_basic.h"
#include "../include/cpu_parallel_processor.h"
#include "../include/search_frame.h"

#define NUM_POINTS_ON_MEAN 5
extern float ___computeMeanHeading(float4 *waypoints, int pos, int size, bool *valid, int width, int height);
extern bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);

class ParallelCheckFeasiblePathVector : public ParallelProcessor
{
    bool _pathFeasible;
    SearchFrame *_searchFrame;
    std::vector<Waypoint> *_path;
    int *_params;
    float *_classCosts;
    int _minDistX;
    int _minDistZ;
    bool _informWaypointIndividualFeasibility;

public:
    ParallelCheckFeasiblePathVector(
        std::vector<Waypoint> *path,
        SearchFrame *searchFrame,
        int *params, float *classCosts,
        int numThreadHandlers,
        int minDistX, int minDistZ) : ParallelProcessor(numThreadHandlers, 1, path->size()),
                                      _path(path), _searchFrame(searchFrame), _params(params),
                                      _classCosts(classCosts), _minDistX(minDistX), _minDistZ(minDistZ), _pathFeasible(true) {}

    void handler(int threadId) override
    {
        if (threadId >= _path->size())
            return;

        const int lower_bound_ego_x = _params[FRAME_PARAM_LOWER_BOUND_X];
        const int lower_bound_ego_z = _params[FRAME_PARAM_LOWER_BOUND_Z];
        const int upper_bound_ego_x = _params[FRAME_PARAM_UPPER_BOUND_X];
        const int upper_bound_ego_z = _params[FRAME_PARAM_UPPER_BOUND_Z];

        int x = (*_path)[threadId].x();
        int z = (*_path)[threadId].z();

        (*_path)[threadId].set_checked_as_feasible(true);

        if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        {
            return;
        }

        const float heading_rad = (*_path)[threadId].heading().rad();
        bool res = __computeFeasibleForAngle(_searchFrame->getPtr(), _params, _classCosts, _minDistX, _minDistZ, x, z, heading_rad);

        if (!res)
        {
            (*_path)[threadId].set_checked_as_feasible(false);
            _pathFeasible = false;
        }
    }

    bool isPathFeasible()
    {
        return _pathFeasible;
    }
};

class ParallelCheckFeasiblePathRaw : public ParallelProcessor
{
    bool _pathFeasible;
    SearchFrame *_searchFrame;
    float *_path;
    int _count;
    int *_params;
    float *_classCosts;
    int _minDistX;
    int _minDistZ;

public:
    ParallelCheckFeasiblePathRaw(
        float *path,
        int count,
        SearchFrame *searchFrame,
        int *params, float *classCosts,
        int numThreadHandlers,
        int minDistX, int minDistZ) : ParallelProcessor(numThreadHandlers, 1, count),
                                      _path(path), _count(count), _searchFrame(searchFrame), _params(params),
                                      _classCosts(classCosts), _minDistX(minDistX), _minDistZ(minDistZ), _pathFeasible(true) {}

    void handler(int threadId) override
    {
        if (threadId >= _count)
            return;

        const int lower_bound_ego_x = _params[FRAME_PARAM_LOWER_BOUND_X];
        const int lower_bound_ego_z = _params[FRAME_PARAM_LOWER_BOUND_Z];
        const int upper_bound_ego_x = _params[FRAME_PARAM_UPPER_BOUND_X];
        const int upper_bound_ego_z = _params[FRAME_PARAM_UPPER_BOUND_Z];

        int pos = 4 * threadId;

        const int x = _path[pos];
        const int z = _path[pos + 1];
        const float heading_rad = _path[pos + 2];
        _path[pos + 3] = 1.0;

        if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        {
            return;
        }
        bool res = __computeFeasibleForAngle(_searchFrame->getPtr(), _params, _classCosts, _minDistX, _minDistZ, x, z, heading_rad);

        if (!res)
        {
            _path[pos + 3] = 0.0;
            _pathFeasible = false;
        }
    }

    bool isPathFeasible()
    {
        return _pathFeasible;
    }
};

bool SearchFrame::checkFeasiblePath(std::vector<Waypoint> *path, int minDistX, int minDistZ)
{
    const int halfDistX = 0.5 * minDistX;
    const int halfDistZ = 0.5 * minDistZ;
#ifdef DRIVELESS_CUDA_ENABLED
    ParallelCheckFeasiblePathVector p(path, this, _params->get(), _classCosts->get(), _numCPUThreadHandlers, halfDistX, halfDistZ);
#else
    ParallelCheckFeasiblePathVector p(path, this, _params.get(), _classCosts.get(), _numCPUThreadHandlers, halfDistX, halfDistZ);
#endif
    p.runAndWait();
    return p.isPathFeasible();
}

bool SearchFrame::checkFeasiblePath(float *path, int count, int minDistX, int minDistZ)
{
    const int halfDistX = 0.5 * minDistX;
    const int halfDistZ = 0.5 * minDistZ;    
#ifdef DRIVELESS_CUDA_ENABLED
    ParallelCheckFeasiblePathRaw p(path, count, this, _params->get(), _classCosts->get(), _numCPUThreadHandlers, halfDistX, halfDistZ);
#else
    ParallelCheckFeasiblePathRaw p(path, count, this, _params.get(), _classCosts.get(), _numCPUThreadHandlers, halfDistX, halfDistZ);
#endif
    p.runAndWait();
    return p.isPathFeasible();
}
