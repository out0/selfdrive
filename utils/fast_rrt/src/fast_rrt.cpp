#include "../include/fast_rrt.h"
#include "../src/kinematic_model.h"

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