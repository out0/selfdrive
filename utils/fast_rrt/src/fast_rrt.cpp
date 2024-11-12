#include "../include/fast_rrt.h"

FastRRT::FastRRT(
    int og_width,
    int og_height,
    float og_real_width_m,
    float og_real_height_m,
    int lower_bound_x,
    int lower_bound_z,
    int upper_bound_x,
    int upper_bound_z,
    float max_steering_angle)
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
