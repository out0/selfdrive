#ifndef H_KINEMATIC_MODEL
#define H_KINEMATIC_MODEL

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

class CurveGenerator
{

public:
    static std::vector<double3> buildCurveWaypoints(
        double3 _center,
        double _rate_w,
        double _rate_h,
        double _lr,
        double _max_steering_angle_deg,
        double3 firstPos,
        double velocity_meters_per_s,
        double steering_angle_deg,
        double path_size,
        bool output_heading_in_degrees = true);

    static std::vector<double3> buildCurveWaypoints(
        double3 _center,
        double _rate_w,
        double _rate_h,
        double _lr,
        double _max_steering_angle_deg,
        double3 start,
        double3 end,
        double velocity_meters_per_s,
        bool output_heading_in_degrees = true);

    static double to_degrees(double angle);

    static double compute_euclidean_dist(double3 &start, double3 &end);
};

class ConstraintsCheckCPU
{
public:
    static bool computeFeasibleForAngle(
        float3 *frame,
        int *classCost,
        int x,
        int z,
        float angle_radians,
        int width,
        int height,
        int min_dist_x,
        int min_dist_z,
        int lower_bound_ego_x,
        int lower_bound_ego_z,
        int upper_bound_ego_x,
        int upper_bound_ego_z);
};

#endif