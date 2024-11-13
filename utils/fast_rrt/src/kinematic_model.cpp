
#include "kinematic_model.h"
#include <stdio.h>


#ifndef PI_F
#define PI_F 3.141592654F
#endif

static float to_radians(float angle)
{
    return (angle * PI_F) / 180;
}

static float to_degrees(float angle)
{
    return (angle * 180) / PI_F;
}

static void convert_to_map_coord(float3 &center, float rate_w, float rate_h, float3 &p)
{
    float x = p.x;
    float z = p.y;

    p.x = (center.x - z) / rate_w;
    p.y = (x - center.y) / rate_h;
}
static void convert_to_waypoint_coord(float3 &center, float rate_w, float rate_h, float3 &p)
{
    float x = p.x;
    float y = p.y;

    p.x = static_cast<int>(floor((center.y + rate_h * y)));
    p.y = static_cast<int>(floor((center.x - rate_w * x)));
}

static float compute_euclidean_dist(float3 &start, float3 &end)
{
    float dx = end.x - start.x;
    float dy = end.y - start.y;
    return sqrtf(dx * dx + dy * dy);
}

static float compute_path_heading(float3 p1, float3 p2)
{
    float dy = p2.y - p1.y;
    float dx = p2.x - p1.x;

    if (dy >= 0 && dx > 0) // Q1
        return atan(dy / dx);
    else if (dy >= 0 && dx < 0) // Q2
        return PI_F - atan(dy / abs(dx));
    else if (dy < 0 && dx > 0) // Q3
        return -atan(abs(dy) / dx);
    else if (dy < 0 && dx < 0) // Q4
        return atan(dy / dx) - PI_F;
    else if (dx == 0 && dy > 0)
        return PI_F / 2;
    else if (dx == 0 && dy < 0)
        return -PI_F / 2;
    return 0.0;
}

static float clip(float val, float min, float max)
{
    if (val < min)
        return min;
    if (val > max)
        return max;
    return val;
}

CurveGenerator::CurveGenerator(float3 center, float rate_w, float rate_h, float lr, float max_steering_angle_deg) {
    _center = center;
    _rate_w = rate_w;
    _rate_h = rate_h;
    _lr = lr;
    _max_steering_angle_deg = max_steering_angle_deg;
}

Memlist<float3> *CurveGenerator::buildCurveWaypoints(float3 start, float velocity_meters_per_s, float steering_angle_deg, float path_size)
{
    float steer = tan(to_radians(steering_angle_deg));

    convert_to_map_coord(_center, _rate_w, _rate_h, start);

    float heading = to_radians(start.z);
    float dt = 0.1;
    int last_x = -1, last_z = -1;
    float ds = velocity_meters_per_s * dt;
    float beta = atan(steer / _lr);

    float x = start.x;
    float y = start.y;

    int max_size = static_cast<int>(floor(path_size)) + 1;

    Memlist<float3> *res = new Memlist<float3>();
    res->data = new float3[max_size];
    res->size = 0;

    while (res->size < path_size)
    {
        x += ds * cosf(heading + beta);
        y += ds * sinf(heading + beta);
        heading += ds * cosf(beta) * steer / (2 * _lr);

        res->data[res->size].x = x;
        res->data[res->size].y = y;
        res->data[res->size].z = to_degrees(heading);

        convert_to_waypoint_coord(_center, _rate_w, _rate_h, res->data[res->size]);

        if (res->data[res->size].x == last_x && res->data[res->size].y == last_z)
            continue;

        res->size++;
    }

    return res;
}

Memlist<float3> *CurveGenerator::buildCurveWaypoints(float3 start, float3 end, float velocity_meters_per_s)
{
    float distance = compute_euclidean_dist(start, end);
    convert_to_map_coord(_center, _rate_w, _rate_h, start);
    convert_to_map_coord(_center, _rate_w, _rate_h, end);
    float dt = 0.1;
    
    Memlist<float3> *res = new Memlist<float3>();
    res->size = 0;
    int last_x = -1, last_z = -1;

    float max_turning_angle = to_radians(_max_steering_angle_deg);
    float heading = to_radians(start.z);

    float path_heading = compute_path_heading(start, end);
    float steering_angle_deg = clip(path_heading - heading, -max_turning_angle, max_turning_angle);
    float ds = velocity_meters_per_s * dt;

    int total_steps = static_cast<int>(round(distance / ds));
    res->data = new float3[total_steps + 1];

    int best_end_pos = -1;
    float best_end_dist = distance;
    float x = start.x;
    float y = start.y;
printf("[CPU] ds=%f\n", ds);

    for (int i = 0; i < total_steps; i++)
    {
        float steer = tan(steering_angle_deg);
        float beta = atan(steer / _lr);

        x += ds * cosf(heading + beta);
        y += ds * sinf(heading + beta);
        heading += ds * cosf(beta) * steer / (2 * _lr);

        if (i >= 9 && i < 20) {
            printf("[CPU %d] x = %f, y=%f, heading=%f\n", i, x, y, heading);
        }

        res->data[res->size].x = x;
        res->data[res->size].y = y;
        res->data[res->size].z = to_degrees(heading);

        path_heading = compute_path_heading(res->data[res->size], end);
        steering_angle_deg = clip(path_heading - heading, -max_turning_angle, max_turning_angle);
        float dist = compute_euclidean_dist(res->data[res->size], end);

        convert_to_waypoint_coord(_center, _rate_w, _rate_h, res->data[res->size]);

        if (res->data[res->size].x == last_x && res->data[res->size].y == last_z)
            continue;

        if (best_end_dist > dist)
        {
            best_end_dist = dist;
            best_end_pos = res->size;
        }

        last_x = static_cast<int>(res->data[res->size].x);
        last_z = static_cast<int>(res->data[res->size].z);

        res->size++;
    }

    
    res->size = best_end_pos + 1;
    return res;
}
