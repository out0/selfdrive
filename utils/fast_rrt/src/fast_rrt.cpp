#include "../include/fast_rrt.h"
#include "../src/kinematic_model.h"
#include <chrono>
#include <cstdlib> // for rand()
#include <ctime>   // for time()
#include <stack>

FastRRT::FastRRT(
    int og_width,
    int og_height,
    double og_real_width_m,
    double og_real_height_m,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_x,
    int lower_bound_z,
    int upper_bound_x,
    int upper_bound_z,
    double max_steering_angle,
    double velocity_m_s)
{
    _og_width = og_width;
    _rw = og_width / og_real_width_m;
    _rh = og_height / og_real_height_m;

    _center.x = static_cast<int>(round(og_width / 2));
    _center.y = static_cast<int>(round(og_height / 2));
    _center.z = 0.0;

    _lr = 0.5 * (lower_bound_z - upper_bound_z) / (og_height / og_real_height_m);
    _max_steering_angle = max_steering_angle;

    _graph = new CudaGraph(
        og_width,
        og_height,
        min_dist_x,
        min_dist_z,
        lower_bound_x,
        lower_bound_z,
        upper_bound_x,
        upper_bound_z,
        _rw,
        _rh,
        max_steering_angle,
        _lr,
        velocity_m_s);
}

FastRRT::~FastRRT()
{
}

std::vector<double3> FastRRT::buildCurveWaypoints(double3 start, double velocity_meters_per_s, double steering_angle_deg, double path_size)
{
    return CurveGenerator::buildCurveWaypoints(
        _center,
        _rw,
        _rh,
        _lr,
        _max_steering_angle,
        start,
        velocity_meters_per_s,
        steering_angle_deg,
        path_size);
}
std::vector<double3> FastRRT::buildCurveWaypoints(double3 start, double3 end, double velocity_meters_per_s)
{
    return CurveGenerator::buildCurveWaypoints(
        _center,
        _rw,
        _rh,
        _lr,
        _max_steering_angle,
        start,
        end,
        velocity_meters_per_s);
}

void FastRRT::testDrawPath(float3 *og, double3 &start, double3 &end)
{
    _graph->drawKinematicPath(og, start, end);
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

void FastRRT::__random_seed()
{
    std::srand(std::time(nullptr)); // Seed with current time
}
int FastRRT::__random_gen(int min, int max)
{
    return min + std::rand() % (max - min + 1);
}

int2 FastRRT::__get_random_node()
{
    int expand_node = __random_gen(0, _node_list.size());
    int x = _node_list[expand_node] / 1000;
    return {x, _node_list[expand_node] - x};
}

int2 FastRRT::__expand_graph()
{
    __random_seed();
    int2 n = __get_random_node();
    int angle = (double)__random_gen(-_max_steering_angle, _max_steering_angle);
    int size = (double)__random_gen(10, RRT_MAX_STEP);

    return _graph->deriveNode(_og, n.x, n.y, angle, size);
}

#define USE_DRIVELESS_CUDAC_OPTIMIZATION

double FastRRT::__compute_distance_to_goal(int2 &node, double3 &goal)
{
    // FAST PRE-COMPUTE FOR OUR PROJECT:

#ifdef USE_DRIVELESS_CUDAC_OPTIMIZATION
    int pos = node.y * _og_width + node.x;
    return _og[pos].y;
#else
    return CurveGenerator::compute_euclidean_dist(goal, node);
#endif
}

bool FastRRT::__rrt_search(int iteraction_time_ms)
{
    auto start = std::chrono::high_resolution_clock::now();

    while (true)
    {
        if (!_search)
            return true;

        if (__check_timeout())
            return true;

        int2 node = __expand_graph();
        if (node.x < 0 || node.y < 0)
            continue;

        _graph->optimizeGraphWithNode(_og, node.x, node.y, RRT_MAX_STEP);

        if (_goal_found)
        {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            if (duration_ms.count() > iteraction_time_ms)
                return false;
        }
        else
        {
            if (__compute_distance_to_goal(node, _goal) <= REACH_DISTANCE)
            {
                _goal_found = true;
                return false;
            }
        }
    }
}

bool FastRRT::__build_path()
{
    int2 n = _graph->find_best_feasible_neighbor(_og,
                                        static_cast<int>(_goal.x),
                                        static_cast<int>(_goal.y),
                                        REACH_DISTANCE);
    if (n.x <= 0 || n.y <= 0) return false;

    std::stack<int2> s;

    // copy from void TestFrame::drawGraph()
}

void FastRRT::__search_executor()
{
    bool loop_search = _search;
    _goal_found = false;
    _path.clear();

    __set_exec_started();
    _graph->add(_start.x, _start.y, _start.z, -1, -1, 0);

    while (loop_search && _search)
    {
        bool timeout = __rrt_search(RRT_SEARCH_ITERATION_EXEC_TIME_ms);

        if (timeout && !_goal_found)
        {
            _search = false; // signals that the planner finished its execution
            return;          // kills the thread
        }

        if (_goal_found)
        {
            __build_path();
            _search = false;
            return;
        }
    }
}
