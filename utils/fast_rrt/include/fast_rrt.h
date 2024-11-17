#ifndef H_FAST_RRT
#define H_FAST_RRT

#include <cuda_runtime.h>
#include <vector>
#include "../src/cuda_graph.h"

class FastRRT
{
private:
    CudaGraph *_graph;
    double3 _center;
    double _rw;
    double _rh;
    double _lr;
    double _max_steering_angle;
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
        double velocity_m_s);
    ~FastRRT();

    void testDrawPath(float3 *og, double3 &start, double3 &end);
};

#endif