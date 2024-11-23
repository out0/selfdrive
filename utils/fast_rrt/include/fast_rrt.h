#ifndef H_FAST_RRT
#define H_FAST_RRT

#include <cuda_runtime.h>
#include <vector>
#include "../src/cuda_graph.h"
#include "../src/cuda_frame.h"
#include <unordered_set>
#include <chrono>

#define RRT_SEARCH_ITERATION_EXEC_TIME_ms 100
#define RRT_MAX_STEP 100
#define REACH_DISTANCE 20.0

class FastRRT
{
private:
    float3 *_og;
    CudaGraph *_graph;
    double3 _center;
    double3 _start;
    double3 _goal;
    int _og_width;
    int _og_height;
    double _rw;
    double _rh;
    double _lr;
    double _max_steering_angle;
    bool _search;
    bool _goal_found;
    long _timeout_ms;
    double _velocity_m_s;

    std::vector<double3> _path;
    std::vector<int> _node_list;
    std::chrono::time_point<std::chrono::high_resolution_clock> _exec_start;

    bool __check_timeout();
    void __set_exec_started();
    long __get_exec_time_ms();

    int2 __expand_graph();
    int2 __get_random_node();
    void __random_seed();
    double __compute_distance_to_goal(int2 &, double3 &);
    int __random_gen(int min, int max);
    bool __build_path();
    bool __rrt_search(int iteraction_time_ms);
    void __add_key(double x, double z);

    void debug_dump_graph(const char *output_file);

    /* data */
public:
    std::vector<double3> buildCurveWaypoints(double3 start, double velocity_meters_per_s, double steering_angle_deg, double path_size);
    std::vector<double3> buildCurveWaypoints(double3 start, double3 end, double velocity_meters_per_s);

    FastRRT(
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
        int timeout_ms);
    ~FastRRT();

    void testDrawPath(float3 *og, double3 &start, double3 &end);
    void setPlanData(float3 *og, double3 start, double3 goal, float velocity_m_s);
    void search();
    void cancel();
    bool isPlanning();
    int getPathSize();
    void getPath(float *result);
};

#endif