#include "../include/fast_rrt.h"

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
    double rw = og_width / og_real_width_m;
    double rh = og_height / og_real_height_m;

    double3 center;
    center.x = static_cast<int>(round(og_width / 2));
    center.y = static_cast<int>(round(og_height / 2));

    double lr = 0.5 * (lower_bound_z - upper_bound_z) / (og_height / og_real_height_m);

    _curveGenerator = new CurveGenerator(
        center,
        rw,
        rh,
        lr,
        max_steering_angle);

    _graph = new CudaGraph(
        og_width,
        og_height,
        min_dist_x,
        min_dist_z,
        lower_bound_x,
        lower_bound_z,
        upper_bound_x,
        upper_bound_z,
        rw,
        rh,
        max_steering_angle,
        lr,
        velocity_m_s);
}

FastRRT::~FastRRT()
{
    delete _curveGenerator;
}

Memlist<double3> *FastRRT::buildCurveWaypoints(double3 start, double velocity_meters_per_s, double steering_angle_deg, double path_size)
{
    return _curveGenerator->buildCurveWaypoints(start, velocity_meters_per_s, steering_angle_deg, path_size);
}
Memlist<double3> *FastRRT::buildCurveWaypoints(double3 start, double3 end, double velocity_meters_per_s)
{
    return _curveGenerator->buildCurveWaypoints(start, end, velocity_meters_per_s);
}

void FastRRT::testDrawPath(float3 *og, double3 &start, double3 &end) {
    _graph->drawKinematicPath(og, start, end);
}