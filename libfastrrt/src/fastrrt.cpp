#include "../include/fastrrt.h"
#include <bits/algorithmfwd.h>

FastRRT::FastRRT(
    int width,
    int height,
    float perceptionWidthSize_m,
    float perceptionHeightSize_m,
    angle maxSteeringAngle,
    float vehicleLength,
    int timeout_ms,
    // std::pair<int, int> minDistance,
    // std::pair<int, int> lowerBound,
    // std::pair<int, int> upperBound,
    // std::vector<float> segmentationClassCost,
    float maxPathSize,
    float distToGoalTolerance) : _graph(CudaGraph(width, height)),
                                 _distToGoalTolerance(distToGoalTolerance),
                                 _timeout_ms(timeout_ms),
                                 _maxPathSize(maxPathSize),
                                 _start(Waypoint(0, 0, angle::rad(0))),
                                 _goal(Waypoint(0, 0, angle::rad(0))),
                                 _hasPlanData(false)
{
    // printf ("Parameters: \n");
    // printf ("width: %d, height: %d\n", width, height);
    // printf ("perception width: %f, height: %f\n", perceptionWidthSize_m, perceptionHeightSize_m);
    // printf ("max steering: deg %f, rad: %f\n", maxSteeringAngle.deg(), maxSteeringAngle.rad());
    // printf ("vehicleLength = %f\n", vehicleLength);
    // printf ("timeout_ms = %d\n", timeout_ms);
    // printf ("minDistance = %d, %d\n", minDistance.first, minDistance.second);
    // printf ("lowerBound = %d, %d\n", lowerBound.first, lowerBound.second);
    // printf ("upperBound = %d, %d\n", upperBound.first, upperBound.second);
    // printf ("maxPathSize = %f\n", maxPathSize);
    // printf ("distToGoalTolerance = %f\n", distToGoalTolerance);

    _graph.setPhysicalParams(perceptionWidthSize_m, perceptionHeightSize_m, maxSteeringAngle, vehicleLength);
    _ptr = nullptr;
}

void FastRRT::__set_exec_started()
{
    _exec_start = std::chrono::high_resolution_clock::now();
}

long FastRRT::__get_exec_time_ms()
{
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - _exec_start);
    return duration_ms.count();
}

bool FastRRT::__check_timeout()
{
    return (_timeout_ms > 0 && __get_exec_time_ms() > _timeout_ms);
}

void FastRRT::setPlanData(SearchFrame &frame, Waypoint start, Waypoint goal, float velocity_m_s, std::pair<int, int> minDistance)
{
    this->_hasPlanData = true;
    this->_start = start;
    this->_goal = goal;
    this->_ptr = frame.getCudaPtr();
    this->_planningVelocity_m_s = velocity_m_s;
    _graph.setSearchParams(minDistance, frame.lowerBound(), frame.upperBound());
    _graph.setClassCosts(frame.getCudaClassCostsPtr(), frame.getClassCount());

    // printf ("_goal.x = %d, _goal.y = %d, _goal.h = %f\n", _goal.x(), _goal.z(), _goal.heading().deg());
}

// extern void exportGraph2(CudaGraph *graph, const char *filename);

void FastRRT::search_init(bool copyIntrinsicCostsFromFrame)
{
    if (!_hasPlanData)
    {
        throw std::runtime_error("unable to initialize planning without planning data");
    }
    __set_exec_started();
    _graph.clear();
    _graph.addStart(_start.x(), _start.z(), _start.heading());
    _last_expanded_node_count = 0;

    // int x = 183, z = 72;
    // printf ("result for %d,%d: z = %.2f\n", x, z, this->_ptr[z * 256 + x].z);
}

void FastRRT::__shrink_search_graph()
{
    std::vector<Waypoint> path = getPlannedPath();
    _graph.clear();
    for (Waypoint &p : path)
        _graph.setType(p.x(), p.z(), GRAPH_TYPE_NODE);
}

bool FastRRT::loop(bool smart)
{
    if (__check_timeout())
    {
        printf("timeout\n");
        return false;
    }

    bool expandFrontier = _last_expanded_node_count >= 100;

    // printf ("_last_expanded_node_count = %d\n", _last_expanded_node_count);

    if (smart)
    {
        _graph.smartExpansion(_ptr, _goal.heading(), _maxPathSize, _planningVelocity_m_s, expandFrontier, _last_expanded_node_count == 0, {_goal.x(), _goal.z()}, _goal.heading());
    }
    else
    {
        _graph.expandTree(_ptr, _goal.heading(), _maxPathSize, _planningVelocity_m_s, expandFrontier, {_goal.x(), _goal.z()}, _goal.heading());
    }

    _last_expanded_node_count = _graph.count(GRAPH_TYPE_TEMP);
    _graph.acceptDerivedNodes();

    if (goalReached())
    {
        // printf ("shrinking graph...\n");
        __shrink_search_graph();
        return false;
    }
    return true;
}

void FastRRT::path_optimize()
{
    if (__check_timeout())
        return;

    _graph.optimizeGraph(_ptr, {_goal.x(), _goal.z()}, _goal.heading(), _distToGoalTolerance, _planningVelocity_m_s);
}

bool FastRRT::goalReached()
{
    int2 goal = {_goal.x(), _goal.z()};
    return _graph.checkGoalReached(_ptr, goal, _goal.heading(), _distToGoalTolerance, TO_RAD * 10);
}

std::vector<Waypoint> FastRRT::getPlannedPath()
{
    std::vector<Waypoint> res;

    if (!_hasPlanData)
        return res;

    if (!goalReached())
        return res;

    // res.push_back(*_goal);
    int2 n = _graph.findBestNode(_ptr, _goal.heading(), _distToGoalTolerance, _goal.x(), _goal.z(), TO_RAD * 10);

    // long i = 0;
    while (n.x != -1 && n.y != -1)
    {
        res.push_back(Waypoint(n.x, n.y, _graph.getHeading(n.x, n.y)));
        n = _graph.getParent(n.x, n.y);

        // if (i >= 1000000)
        // {
        //     printf("looping too much (%d, %d) i = %ld\n", n.x, n.y, i);
        //     //break;
        // }
        // i++;
    }

    std::reverse(res.begin(), res.end());

    // for (auto p : res) {
    //     printf("(%d, %d) - ", p.x(), p.z());
    // }
    // printf("\n");

    return res;
}

extern std::vector<Waypoint> interpolate(std::vector<Waypoint> &path, int width, int height);

std::vector<Waypoint> FastRRT::interpolatePlannedPath()
{
    auto v = getPlannedPath();
    return interpolate(v, _graph.width(), _graph.height());
}

std::vector<Waypoint> FastRRT::interpolatePlannedPath(std::vector<Waypoint> path)
{
    return interpolate(path, _graph.width(), _graph.height());
}

std::vector<int3> FastRRT::exportGraphNodes()
{
    return _graph.listAll();
}

extern std::vector<Waypoint> interpolateHermiteCurve(int width, int height, Waypoint p1, Waypoint p2);

std::vector<Waypoint> FastRRT::idealGeometryCurveNoObstacles(Waypoint goal)
{
    float3 *start = _graph.getCoordinateStart();
    return interpolateHermiteCurve(
        _graph.width(),
        _graph.height(),
        Waypoint(
            static_cast<int>((*start).x),
            static_cast<int>((*start).y),
            angle::rad(static_cast<float>((*start).z))),
        goal);
}

void FastRRT::__computeGraphRegionDensity()
{
    _graph.__computeGraphRegionDensity();
}