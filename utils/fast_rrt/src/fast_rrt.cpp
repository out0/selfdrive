#include "../include/fast_rrt.h"

FastRRT::FastRRT(
    int og_width,
    int og_height,
    float og_real_width_m,
    float og_real_height_m,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_x,
    int lower_bound_z,
    int upper_bound_x,
    int upper_bound_z,
    float max_steering_angle,
    float velocity_m_s)
{
    float rw = og_width / og_real_width_m;
    float rh = og_height / og_real_height_m;

    float3 center;
    center.x = static_cast<int>(round(og_width / 2));
    center.y = static_cast<int>(round(og_height / 2));

    float lr = 0.5 * (lower_bound_z - upper_bound_z) / (og_height / og_real_height_m);

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

Memlist<float3> *FastRRT::buildCurveWaypoints(float3 start, float velocity_meters_per_s, float steering_angle_deg, float path_size)
{
    return _curveGenerator->buildCurveWaypoints(start, velocity_meters_per_s, steering_angle_deg, path_size);
}
Memlist<float3> *FastRRT::buildCurveWaypoints(float3 start, float3 end, float velocity_meters_per_s)
{
    return _curveGenerator->buildCurveWaypoints(start, end, velocity_meters_per_s);
}

void FastRRT::testDrawPath(float3 *og, float3 &start, float3 &end) {
    _graph->drawKinematicPath(og, start, end);
}