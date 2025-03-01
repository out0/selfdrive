#include "../include/fast_rrt.h"
#include "../src/kinematic_model.h"
#include <cstdlib> // for rand()
#include <ctime>   // for time()
#include <stack>
#include "../src/cuda_basic.h"
#include "../src/class_def.h"

// #define DEBUG_DUMP_GRAPH

#ifdef DEBUG_DUMP_GRAPH
#include <opencv2/opencv.hpp>

void show_point(cv::Mat &mat, double x, double z, int r, int g, int b)
{
    if (x < 0 || x >= mat.cols)
        return;
    if (z < 0 || z >= mat.rows)
        return;
    cv::Vec3b &pixel = mat.at<cv::Vec3b>(static_cast<int>(z), static_cast<int>(x));
    pixel[0] = r;
    pixel[1] = g;
    pixel[2] = b;
}

void FastRRT::debug_dump_graph(const char *output_file)
{
    cv::Mat image(_og_height, _og_width, CV_8UC3);

    for (int z = 0; z < _og_height; z++)
        for (int x = 0; x < _og_width; x++)
        {
            int c = _og[z * _og_width + x].x;
            cv::Vec3b &pixel = image.at<cv::Vec3b>(z, x);
            pixel[0] = segmentationClassColors[c][0];
            pixel[1] = segmentationClassColors[c][1];
            pixel[2] = segmentationClassColors[c][2];
        }

    unsigned int count = _graph->count();

    printf("dumping %d points\n", count);

    std::vector<int2> points = _graph->list();

    double3 end;

    for (int2 p : points)
    {
        double3 start = _graph->getParent(p.x, p.y);
        if (start.x < 0 || start.y < 0)
            continue; // initial point should be ignored because it has no parent

        end.x = p.x;
        end.y = p.y;
        end.z = start.z;

        std::vector<double3> curve = CurveGenerator::buildCurveWaypoints(_center, _rw, _rh, _lr, _max_steering_angle, start, end, _velocity_m_s);

        for (double3 point : curve)
        {
            show_point(image, point.x, point.y, 255, 255, 255);
        }

        show_point(image, start.x, start.y, 0, 0, 255);
        show_point(image, end.x, end.y, 0, 0, 255);
    }

    // Save the image to verify the change
    cv::imwrite(output_file, image);
}
#endif

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
    int timeout_ms)
{
    _og_width = og_width;
    _og_height = og_height;
    _rw = og_width / og_real_width_m;
    _rh = og_height / og_real_height_m;
    _timeout_ms = static_cast<long>(timeout_ms);

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
        _lr);
}

FastRRT::~FastRRT()
{
    delete _graph;
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

void FastRRT::__add_key(double x, double z)
{
    int key = 1000 * static_cast<int>(z) + static_cast<int>(x);
    _node_list.push_back(key);
}

int FastRRT::__random_gen(int min, int max)
{
    return min + std::rand() % (max - min + 1);
}

int2 FastRRT::__expand_graph()
{
    int2 n = _graph->get_random_node();
    int angle = (double)__random_gen(-_max_steering_angle, _max_steering_angle);
    int size = (double)__random_gen(10, RRT_MAX_STEP);

    int2 n_final = _graph->deriveNode(_og, n.x, n.y, angle, size);

    if (n_final.x < 0 || n_final.y < 0)
        return n_final;

    __add_key(n_final.x, n_final.y);
    return n_final;
}

// #define USE_DRIVELESS_CUDAC_OPTIMIZATION

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

#ifdef DEBUG_DUMP_GRAPH
        debug_dump_graph("curve_debug.png");
#endif

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
    int2 node = _graph->find_best_feasible_neighbor(_og,
                                                    static_cast<int>(_goal.x),
                                                    static_cast<int>(_goal.y),
                                                    REACH_DISTANCE);

    if (node.x <= 0 || node.y <= 0)
        return false;

    double3 start, end;
    end.x = node.x;
    end.y = node.y;
    end.z = 0.0;

    _path.clear();

    while (end.x >= 0 && end.y >= 0)
    {
        double3 start = _graph->getParent(end.x, end.y);

        if (start.x >= 0 && start.y >= 0)
        {
            std::vector<double3> waypoints = CurveGenerator::buildCurveWaypoints(_center, _rw, _rh, _lr, _max_steering_angle, start, end, _velocity_m_s, true);
            _path.insert(_path.end(), waypoints.begin(), waypoints.end());
        }

        end.x = start.x;
        end.y = start.y;
        end.z = start.z;
    }
    return true;
}
void FastRRT::cancel()
{
    _search = false;
}
void FastRRT::search()
{
    bool loop_search = _search;
    _goal_found = false;

    _graph->clear();
    _path.clear();

    __set_exec_started();
    _graph->add(_start.x, _start.y, _start.z, -1, -1, 0);
    __add_key(_start.x, _start.y);

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
void FastRRT::setPlanData(float3 *og, double3 start, double3 goal, float velocity_m_s)
{
    _graph->clear();
    _og = og;
    _start = start;
    _goal = goal;
    _velocity_m_s = static_cast<double>(velocity_m_s);
    _search = true;
    _goal_found = false;
}

bool FastRRT::isPlanning()
{
    return _search;
}

int FastRRT::getPathSize()
{
    if (_search)
        return 0;
    return _path.size();
}
std::vector<double3> & FastRRT::getPath()
{
    return _path;
}
void FastRRT::copyPathTo(float *result)
{
    int size = getPathSize();
    for (int i = 0; i < size; i++)
    {
        int pos = 3 * i;
        result[pos] = static_cast<float>(_path[i].x);
        result[pos + 1] = static_cast<float>(_path[i].y);
        result[pos + 2] = static_cast<float>(_path[i].z);
    }
}