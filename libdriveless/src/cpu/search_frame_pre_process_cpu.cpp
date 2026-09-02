#include "../../include/search_frame.h"
#include "../../include/cpu_parallel_processor.h"
#include "../../include/cuda_basic.h"
#include "../search_frame_params.h"
#include <stdexcept>

extern bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);

extern const float TRAVERSABILITY_ANGLES[];
extern const float H_TRAVERSABILITY_ANGLES[];
extern const int TRAVERSABILITY_BITS[];
extern const int H_TRAVERSABILITY_BITS[];

extern std::pair<int, int> __checkTraversableAngleBitPairCheck(float heading_rad);
extern bool CHECK_OUT_BOUNDARIES(int width, int height, int x, int z);
extern void setObstacle(float3 *frame, int width, int height, int x, int z);
extern bool isObstacle(float3 *frame, float *classCosts, int width, int height, int x, int z);
extern void propagateObstacleInRegion(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start);
extern void propagateObstacleLeft(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start);
extern void propagateObstacleRight(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start);
extern void propagateObstacleTop(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start);
extern void propagateObstacleBottom(float3 *frame, const int width, const int height, const int minDistance, int x_start, int z_start);
extern void propagateMinDistance(float3 *frame, float *classCosts, const int width, const int height, const int minDistance, int pos, int x, int z);
extern void count_obstacle_in_search_zones(float3 *frame, float *classCosts, int *search_params, uint4 *search_zone_info, int pos);

class SafeDistancePrepareProcessor : public ParallelProcessor
{
private:
    float3 *_frame;
    float *_classCosts;
    int *_searchSpaceParams;
    int _half_minDist_px;
    int _maxId;

public:
    SafeDistancePrepareProcessor(float3 *frame, float *classCosts,
                                 int *searchSpaceParams, int half_minDist_px, int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, searchSpaceParams[FRAME_PARAM_HEIGHT], searchSpaceParams[FRAME_PARAM_WIDTH]),
                                                                                                            _frame(frame), _classCosts(classCosts), _searchSpaceParams(searchSpaceParams), _half_minDist_px(half_minDist_px)
    {
        _maxId = _searchSpaceParams[FRAME_PARAM_HEIGHT] * _searchSpaceParams[FRAME_PARAM_WIDTH];
    }

    void handler(int pos) override
    {
        if (pos >= _maxId)
            return;

        int width = _searchSpaceParams[FRAME_PARAM_WIDTH];
        int height = _searchSpaceParams[FRAME_PARAM_HEIGHT];
        int lower_bound_ego_x = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_X];
        int lower_bound_ego_z = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_Z];
        int upper_bound_ego_x = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_X];
        int upper_bound_ego_z = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_Z];

        int z = pos / width;
        int x = pos - z * width;

        // turns on the obstacle propagation-based traversability bit and off the angle-based bits (0001 0000 0000)
        // because the obstacle propagation-based works by propagating obstacles as turning bits off, while the
        // angle-based check works by checking each angle and turning the respective bit on as it is traversable.
        _frame[pos].z = 256.0;

        if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
            return;

        const int nodeClass = TO_INT(_frame[pos].x);
        if (_classCosts[nodeClass] < 0)
        {
            _frame[pos].z = 0x0;
        }
    }
};

class SafeDistanceObstacleExpansionBasedProcessor : public ParallelProcessor
{
private:
    float3 *_frame;
    float *_classCosts;
    int *_searchSpaceParams;
    int _half_minDist_px;
    int _maxId;
    uint4 *_search_zone_info;

public:
    SafeDistanceObstacleExpansionBasedProcessor(float3 *frame, uint4 *search_zone_info, float *classCosts,
                                                int *searchSpaceParams, int half_minDist_px, int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, searchSpaceParams[FRAME_PARAM_HEIGHT], searchSpaceParams[FRAME_PARAM_WIDTH]),
                                                                                                                           _frame(frame), _search_zone_info(search_zone_info), _classCosts(classCosts), _searchSpaceParams(searchSpaceParams), _half_minDist_px(half_minDist_px)
    {
        _maxId = _searchSpaceParams[FRAME_PARAM_HEIGHT] * _searchSpaceParams[FRAME_PARAM_WIDTH];
    }

    void handler(int pos) override
    {
        if (pos >= _maxId)
            return;
        int width = _searchSpaceParams[FRAME_PARAM_WIDTH];
        int height = _searchSpaceParams[FRAME_PARAM_HEIGHT];
        int lower_bound_ego_x = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_X];
        int lower_bound_ego_z = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_Z];
        int upper_bound_ego_x = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_X];
        int upper_bound_ego_z = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_Z];

        int z = pos / width;
        int x = pos - z * width;

        if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
            return;

        const int nodeClass = TO_INT(_frame[pos].x);

        if (_classCosts[nodeClass] < 0)
        {
            // printf("[CUDA] pos %d, %d will propagate distance %d\n", x, z, half_minDist_px);
            propagateMinDistance(_frame, _classCosts, width, height, _half_minDist_px, pos, x, z);
            count_obstacle_in_search_zones(_frame, _classCosts, _searchSpaceParams, _search_zone_info, pos);
        }
    }
};

class SafeDistanceVectorBasedProcessor : public ParallelProcessor
{
private:
    float3 *_frame;
    float *_classCosts;
    int *_searchSpaceParams;
    int _minDistX;
    int _minDistZ;
    int _maxId;

public:
    SafeDistanceVectorBasedProcessor(float3 *frame, float *classCosts,
                                     int *searchSpaceParams, int minDistX, int minDistZ, int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, searchSpaceParams[FRAME_PARAM_HEIGHT], searchSpaceParams[FRAME_PARAM_WIDTH]),
                                                                                                                       _frame(frame), _classCosts(classCosts), _searchSpaceParams(searchSpaceParams), _minDistX(minDistX), _minDistZ(minDistZ)
    {
        _maxId = _searchSpaceParams[FRAME_PARAM_HEIGHT] * _searchSpaceParams[FRAME_PARAM_WIDTH];
    }

    void handler(int pos) override
    {
        if (pos >= _maxId)
            return;

        int width = _searchSpaceParams[FRAME_PARAM_WIDTH];
        int height = _searchSpaceParams[FRAME_PARAM_HEIGHT];
        int lower_bound_ego_x = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_X];
        int lower_bound_ego_z = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_Z];
        int upper_bound_ego_x = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_X];
        int upper_bound_ego_z = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_Z];

        int z = pos / width;
        int x = pos - z * width;

        if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
        {
            _frame[pos].z = 0.0 + (TO_INT(_frame[pos].z) | 0xff);
            return;
        }

        const int nodeClass = TO_INT(_frame[pos].x);

        if (_classCosts[nodeClass] < 0)
            return;

        int v = 0;
        for (int i = 0; i < 8; i++)
        {
            if (__computeFeasibleForAngle(_frame, _searchSpaceParams, _classCosts, _minDistX, _minDistZ, x, z, TRAVERSABILITY_ANGLES[i]))
                v = v | TRAVERSABILITY_BITS[i];
        }

        _frame[pos].z = 0.0 + (TO_INT(_frame[pos].z) | v);
    }
};

void SearchFrame::processSafeDistanceZone(std::pair<int, int> minDistance, bool computeVectorized)
{
    _params.get()[FRAME_PARAM_MIN_DIST_X] = 0.5 * minDistance.first;
    _params.get()[FRAME_PARAM_MIN_DIST_Z] = 0.5 * minDistance.second;

    int min_x = _params.get()[FRAME_PARAM_MIN_DIST_X];
    int min_z = _params.get()[FRAME_PARAM_MIN_DIST_Z];

    int minDist_px = TO_INT(sqrtf(min_x * min_x + min_z * min_z));

    //__CUDA_safe_distance_prepare<<<numBlocks, THREADS_IN_BLOCK>>>(getPtr(), _classCosts->get(), _params->get(), minDist_px);
    // CUDA(cudaDeviceSynchronize());
    SafeDistancePrepareProcessor(getPtr(), _classCosts.get(), _params.get(), minDist_px).runAndWait();

    // __CUDA_safe_distance_obstacle_expansion_based<<<numBlocks, THREADS_IN_BLOCK>>>(getPtr(), _classCosts->get(), _params->get(), minDist_px);
    // CUDA(cudaDeviceSynchronize());
    SafeDistanceObstacleExpansionBasedProcessor(getPtr(), _search_zone_info->getPtr(), _classCosts.get(), _params.get(), minDist_px).runAndWait();

    _safeZoneChecked = true;

    if (computeVectorized)
    {
        //__CUDA_safe_distance_vector_based<<<numBlocks, THREADS_IN_BLOCK>>>(getPtr(), _classCosts->get(), _params->get(), min_x, min_z);
        // CUDA(cudaDeviceSynchronize());
        SafeDistanceVectorBasedProcessor(getPtr(), _classCosts.get(), _params.get(), min_x, min_z).runAndWait();
        _safeZoneVectorialChecked = true;
    }
}

class DistanceToGoalProcessor : public ParallelProcessor
{
private:
    float3 *_frame;
    float *_classCosts;
    int *_searchSpaceParams;
    int _goal_x;
    int _goal_z;
    int _maxId;

public:
    DistanceToGoalProcessor(float3 *frame, float *classCosts,
                            int *searchSpaceParams, int goal_x, int goal_z, int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, searchSpaceParams[FRAME_PARAM_HEIGHT], searchSpaceParams[FRAME_PARAM_WIDTH]),
                                                                                                          _frame(frame), _classCosts(classCosts), _searchSpaceParams(searchSpaceParams), _goal_x(goal_x), _goal_z(goal_z)
    {
        _maxId = _searchSpaceParams[FRAME_PARAM_HEIGHT] * _searchSpaceParams[FRAME_PARAM_WIDTH];
    }

    void handler(int pos) override
    {
        if (pos >= _maxId)
            return;

        int width = _searchSpaceParams[FRAME_PARAM_WIDTH];
        int height = _searchSpaceParams[FRAME_PARAM_HEIGHT];
        int lower_bound_ego_x = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_X];
        int lower_bound_ego_z = _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_Z];
        int upper_bound_ego_x = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_X];
        int upper_bound_ego_z = _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_Z];

        int z = pos / width;
        int x = pos - z * width;

        if (x < lower_bound_ego_x || x > upper_bound_ego_x || z < upper_bound_ego_z || z > lower_bound_ego_z)
        {
            const int nodeClass = TO_INT(_frame[pos].x);
            if (_classCosts[nodeClass] < 0)
            {

                _frame[pos].y = 999999999;
                return;
            }
        }

        float dx = _goal_x - x;
        float dz = _goal_z - z;

        _frame[pos].y = sqrtf(dx * dx + dz * dz);
    }
};

void SearchFrame::processDistanceToGoal(int x, int z)
{
    if (_classCosts.get() == nullptr)
    {
        throw std::runtime_error("Class costs were not set. Please set costs before processing distance to goal.");
    }

    // __CUDA_distance_to_goal<<<numBlocks, THREADS_IN_BLOCK>>>(getPtr(), _classCosts->get(), _params->get(), x, z);
    // CUDA(cudaDeviceSynchronize());

    DistanceToGoalProcessor(getPtr(), _classCosts.get(), _params.get(), x, z).runAndWait();
    _distanceToGoalProcessed = true;
}

float SearchFrame::getDistanceToGoal(int x, int z)
{
    float3 *ptr = getPtr();
    return ptr[z * width() + x].y;
}

std::pair<int4, int> computeICR(float *physical_params, Waypoint p1, bool invert_angle)
{
    const float max_steering_angle = physical_params[PHYSICAL_PARAM_MAX_STEERING_RAD];
    const float wheelbase_px = physical_params[PHYSICAL_PARAM_WHEELBASE_PX];
    const float steer = tanf(max_steering_angle);
    const float beta = atanf(steer / 2);
    float curvature = cosf(beta) * steer / (2 * wheelbase_px);
    if (curvature < 0)
        curvature = -1 * curvature;
    
    const float R = 1 / curvature;
    const float heading = invert_angle ? p1.heading().rad() + PI : p1.heading().rad();

    int4 coordinates;

    coordinates.x = p1.x() + R * cosf(heading + beta);
    coordinates.y = p1.z() + R * sinf(heading + beta);
    coordinates.z = p1.x() - R * cosf(heading - beta);
    coordinates.w = p1.z() - R * sinf(heading - beta);

    return {coordinates, TO_INT(R)};
}

class KinematicExclusionAreasProcessor : public ParallelProcessor
{
private:
    float3 *_frame;
    float *_classCosts;
    int *_searchSpaceParams;
    int4 _origin;
    int4 _goal;
    float _Rsqd;
    int _maxId;

public:
    KinematicExclusionAreasProcessor(float3 *frame, int *searchSpaceParams, int4 origin, int4 goal, float Rsqd, int numThreadHandlers = 12) : ParallelProcessor(numThreadHandlers, searchSpaceParams[FRAME_PARAM_HEIGHT], searchSpaceParams[FRAME_PARAM_WIDTH]),
                                                                                                                                              _frame(frame), _searchSpaceParams(searchSpaceParams), _origin(origin), _goal(goal), _Rsqd(Rsqd)
    {
        _maxId = _searchSpaceParams[FRAME_PARAM_HEIGHT] * _searchSpaceParams[FRAME_PARAM_WIDTH];
    }

    void handler(int pos) override
    {
        if (pos >= _maxId)
            return;

        const int width = _searchSpaceParams[FRAME_PARAM_WIDTH];
        const int z = pos / width;
        const int x = pos - z * width;

        const int dx1 = _origin.x - x;
        const int dz1 = _origin.y - z;
        const int dx2 = _origin.z - x;
        const int dz2 = _origin.w - z;

        if ((dx1 * dx1 + dz1 * dz1) < _Rsqd)
        {
            _frame[pos].z = 0.0;
            return;
        }
        if ((dx2 * dx2 + dz2 * dz2) < _Rsqd)
        {
            _frame[pos].z = 0.0;
            return;
        }

        const int dx3 = _goal.x - x;
        const int dz3 = _goal.y - z;
        const int dx4 = _goal.z - x;
        const int dz4 = _goal.w - z;

        if ((dx3 * dx3 + dz3 * dz3) < _Rsqd)
        {
            _frame[pos].z = 0.0;
            return;
        }
        if ((dx4 * dx4 + dz4 * dz4) < _Rsqd)
        {
            _frame[pos].z = 0.0;
            return;
        }
    }
};

void SearchFrame::processKinematicExclusionAreas(Waypoint origin, Waypoint goal)
{
    if (getPhysicalParamsPtr() == nullptr)
    {
        throw std::invalid_argument("Can only execute processKinematicExclusionAreas with physical params set\n");
    }

    int size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    std::pair<int4, float> icr_origin = computeICR(getPhysicalParamsPtr(), origin, false);
    std::pair<int4, float> icr_goal = computeICR(getPhysicalParamsPtr(), goal, true);

    KinematicExclusionAreasProcessor(getPtr(), _params.get(), icr_origin.first, icr_goal.first, icr_goal.second).runAndWait();
}
