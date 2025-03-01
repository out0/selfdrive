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
    float distToGoalTolerance) : _graph(CudaGraph(width, height)), _distToGoalTolerance(distToGoalTolerance), _timeout_ms(timeout_ms), _maxPathSize(maxPathSize)
{
    _graph.setPhysicalParams(perceptionWidthSize_m, perceptionHeightSize_m, maxSteeringAngle, vehicleLength);
    _graph.setSearchParams(minDistance, lowerBound, upperBound);
    _graph.setClassCosts((int*)segmentationClassColors, 29);
    _goal_found = false;
    _search = false;
    _goal = nullptr;
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

void FastRRT::setPlanData(cudaPtr ptr, Waypoint *goal, float velocity_m_s)
{
    this->_goal = goal;
    this->_ptr = ptr;
    this->_planningVelocity_m_s = velocity_m_s;
}

// extern void exportGraph2(CudaGraph *graph, const char *filename);

void FastRRT::run()
{
    if (_goal == nullptr)
        return;

    _search = true;

    __set_exec_started();
    _graph.clear();
    _graph.addStart();
    // exportGraph2(&_graph, "debug.png");

    while (!__check_timeout() && _search)
    {
        _graph.derivateNode(_ptr, _goal->heading(), _maxPathSize, _planningVelocity_m_s);
        // exportGraph2(&_graph, "debug.png");
        //_graph.optimizeGraph(_frame, MAX_PATH_SIZE, _planningVelocity_m_s);
        // exportGraph2(&_graph, "debug.png");
        _graph.acceptDerivatedNodes();
        // exportGraph2(&_graph, "debug.png");

        if (goalReached())
        {
            std::vector<Waypoint> path = getPlannedPath();
            //printf("path found with %ld nodes\n", path.size());

            _graph.clear();

            for (Waypoint &p : path)
                _graph.setType(p.x(), p.z(), GRAPH_TYPE_NODE);

            // exportGraph2(&_graph, "debug.png");
            _search = false;
        }
    }
}

void FastRRT::optimize()
{
    if (!goalReached())
        return;

    _search = true;
    //__set_exec_started();

    while (!__check_timeout() && _search)
    {
        _graph.derivateNode(_ptr, _goal->heading(), _maxPathSize, _planningVelocity_m_s);
        _graph.optimizeGraph(_ptr, _goal->heading(), _maxPathSize, _planningVelocity_m_s);
        _graph.acceptDerivatedNodes();
    }

    _search = false;
}

void FastRRT::cancel()
{
    _search = false;
}

bool FastRRT::goalReached()
{
    if (_goal == nullptr)
        return false;
    int2 goal = {_goal->x(), _goal->z()};
    return _graph.checkGoalReached(_ptr, goal, _goal->heading(), _distToGoalTolerance);
}

std::vector<Waypoint> FastRRT::getPlannedPath()
{
    std::vector<Waypoint> res;

    if (!goalReached())
        return res;

    // res.push_back(*_goal);
    int2 n = _graph.findBestNode(_ptr, _goal->heading(), _distToGoalTolerance, _goal->x(), _goal->z());

    while (n.x != -1 && n.y != -1)
    {
        res.push_back(Waypoint(n.x, n.y, _graph.getHeading(n.x, n.y)));
        n = _graph.getParent(n.x, n.y);
    }

    std::reverse(res.begin(), res.end());
    return res;
}