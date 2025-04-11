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
    std::pair<int, int> minDistance,
    std::pair<int, int> lowerBound,
    std::pair<int, int> upperBound,
    float maxPathSize,
    float distToGoalTolerance) : 
        _graph(CudaGraph(width, height)), 
        _distToGoalTolerance(distToGoalTolerance), 
        _timeout_ms(timeout_ms), 
        _maxPathSize(maxPathSize), 
        _start(Waypoint(0, 0, angle::rad(0))),
        _goal(Waypoint(0, 0, angle::rad(0)))
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
    _graph.setSearchParams(minDistance, lowerBound, upperBound);
    _graph.setClassCosts((int *)segmentationClassCost, 29);
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

void FastRRT::setPlanData(cudaPtr ptr, Waypoint start, Waypoint goal, float velocity_m_s)
{
    this->_start = start;
    this->_goal = goal;
    this->_ptr = ptr;
    this->_planningVelocity_m_s = velocity_m_s;

    //printf ("_goal.x = %d, _goal.y = %d, _goal.h = %f\n", _goal.x(), _goal.z(), _goal.heading().deg());
}

// extern void exportGraph2(CudaGraph *graph, const char *filename);

void FastRRT::search_init()
{
    __set_exec_started();
    _graph.clear();
    _graph.addStart(_start.x(), _start.z(), _start.heading());
}

void FastRRT::__shrink_search_graph()
{
    std::vector<Waypoint> path = getPlannedPath();
    _graph.clear();
    for (Waypoint &p : path)
        _graph.setType(p.x(), p.z(), GRAPH_TYPE_NODE);
}

bool FastRRT::loop()
{
    if (__check_timeout()) {
        printf ("timeout\n");
        return false;
    }

    _graph.expandTree(_ptr, _goal.heading(), _maxPathSize, _planningVelocity_m_s, true);
    //printf ("new nodes = %d\n", _graph.count(GRAPH_TYPE_TEMP));
    
    if (!_graph.checkNewNodesAddedOnTreeExpansion()) {
        _graph.expandTree(_ptr, _goal.heading(), _maxPathSize, _planningVelocity_m_s, false);
        //printf ("full expandTree\n");
    }

    _graph.acceptDerivatedNodes();

    if (goalReached())
    {
        __shrink_search_graph();
        return false;
    }
    return true;
}

bool FastRRT::loop_optimize()
{
    if (__check_timeout())
        return false;

    _graph.optimizeGraph(_ptr, _goal.heading(), _maxPathSize, _planningVelocity_m_s);
    __shrink_search_graph();
    return true;
}


bool FastRRT::goalReached()
{
    //printf ("goalReached: _goal.x = %d, _goal.y = %d, _goal.h = %f\n", _goal.x(), _goal.z(), _goal.heading().deg());
    int2 goal = {_goal.x(), _goal.z()};
    return _graph.checkGoalReached(_ptr, goal, _goal.heading(), _distToGoalTolerance);
}

std::vector<Waypoint> FastRRT::getPlannedPath()
{
    std::vector<Waypoint> res;

    if (!goalReached())
        return res;

    // res.push_back(*_goal);
    int2 n = _graph.findBestNode(_ptr, _goal.heading(), _distToGoalTolerance, _goal.x(), _goal.z());

    while (n.x != -1 && n.y != -1)
    {
        res.push_back(Waypoint(n.x, n.y, _graph.getHeading(n.x, n.y)));
        n = _graph.getParent(n.x, n.y);
    }

    std::reverse(res.begin(), res.end());
    return res;
}


extern std::vector<Waypoint> interpolate(std::vector<Waypoint>& path, int width, int height);

std::vector<Waypoint> FastRRT::interpolatePlannedPath()
{
    auto v = getPlannedPath();
    return interpolate(v, _graph.width(), _graph.height());
}

std::vector<Waypoint> FastRRT::interpolatePlannedPath(std::vector<Waypoint> path)
{
    return interpolate(path, _graph.width(), _graph.height());
}

std::vector<int3> FastRRT::exportGraphNodes() {
   return _graph.listAll();
}


extern std::vector<Waypoint> interpolateHermiteCurve(int width, int height, Waypoint p1, Waypoint p2);

std::vector<Waypoint> FastRRT::idealGeometryCurveNoObstacles(Waypoint goal) {
    int2 center = _graph.getCenter();
    return interpolateHermiteCurve(_graph.width(), _graph.height(), Waypoint(center.x, center.y, angle::deg(0)), goal);
}
