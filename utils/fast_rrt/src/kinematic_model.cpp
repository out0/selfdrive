
#include "kinematic_model.h"
#include <stdio.h>

#ifndef PI
#define PI 3.1415926535897931e+0
#define PI_2 1.5707963267948966e+0
#endif

double CurveGenerator::to_radians(double angle)
{
    return (angle * PI) / 180;
}

double CurveGenerator::to_degrees(double angle)
{
    return (angle * 180) / PI;
}

static void convert_to_map_coord(double3 &center, double rate_w, double rate_h, double3 &p)
{
    double x = p.x;
    double z = p.y;

    p.x = (center.x - z) / rate_w;
    p.y = (x - center.y) / rate_h;
}
static void convert_to_waypoint_coord(double3 &center, double rate_w, double rate_h, double3 &p)
{
    double x = p.x;
    double y = p.y;

    p.x = static_cast<int>(round((center.y + rate_h * y)));
    p.y = static_cast<int>(round((center.x - rate_w * x)));
}

double CurveGenerator::compute_euclidean_dist(double3 &start, double3 &end)
{
    double dx = end.x - start.x;
    double dy = end.y - start.y;
    return sqrt(dx * dx + dy * dy);
}

static double compute_path_heading(double3 p1, double3 p2)
{
    double dy = p2.y - p1.y;
    double dx = p2.x - p1.x;

    if (dy >= 0 && dx > 0) // Q1
        return atan(dy / dx);
    else if (dy >= 0 && dx < 0) // Q2
        return PI - atan(dy / abs(dx));
    else if (dy < 0 && dx > 0) // Q3
        return -atan(abs(dy) / dx);
    else if (dy < 0 && dx < 0) // Q4
        return atan(dy / dx) - PI;
    else if (dx == 0 && dy > 0)
        return PI_2;
    else if (dx == 0 && dy < 0)
        return -PI_2;
    return 0.0;
}

static double clip(double val, double min, double max)
{
    if (val < min)
        return min;
    if (val > max)
        return max;
    return val;
}

std::vector<double3> CurveGenerator::buildCurveWaypoints(
    double3 _center,
    double _rate_w,
    double _rate_h,
    double _lr,
    double _max_steering_angle_deg,
    double3 start,
    double velocity_meters_per_s,
    double steering_angle_deg,
    double path_size,
    bool output_heading_in_degrees)
{
    double steer = tan(to_radians(steering_angle_deg));

    convert_to_map_coord(_center, _rate_w, _rate_h, start);

    double heading = to_radians(start.z);
    double dt = 0.1;
    int last_x = -1, last_z = -1;
    double ds = velocity_meters_per_s * dt;
    double beta = atan(steer / _lr);

    double x = start.x;
    double y = start.y;

    int max_size = static_cast<int>(round(path_size)) + 1;

    std::vector<double3> res;
    res.reserve(max_size % 100);

    while (res.size() < path_size)
    {
        x += ds * cos(heading + beta);
        y += ds * sin(heading + beta);
        heading += ds * cos(beta) * steer / (2 * _lr);

        double3 p;
        p.x = x;
        p.y = y;

        if (output_heading_in_degrees)
            p.z = to_degrees(heading);
        else
            p.z = heading;

        convert_to_waypoint_coord(_center, _rate_w, _rate_h, p);

        if (p.x == last_x && p.y == last_z)
            continue;

        res.push_back(p);
    }

    return res;
}

std::vector<double3> CurveGenerator::buildCurveWaypoints(
    double3 _center,
    double _rate_w,
    double _rate_h,
    double _lr,
    double _max_steering_angle_deg,
    double3 start,
    double3 end,
    double velocity_meters_per_s,
    bool output_heading_in_degrees)
{
    double distance = compute_euclidean_dist(start, end);
    convert_to_map_coord(_center, _rate_w, _rate_h, start);
    convert_to_map_coord(_center, _rate_w, _rate_h, end);
    double dt = 0.1;

    int last_x = -1, last_y = -1;

    double max_turning_angle = to_radians(_max_steering_angle_deg);
    double heading = to_radians(start.z);

    double path_heading = compute_path_heading(start, end);
    double steering_angle_deg = clip(path_heading - heading, -max_turning_angle, max_turning_angle);
    double ds = velocity_meters_per_s * dt;

    int total_steps = static_cast<int>(round(distance / ds));

    std::vector<double3> res;
    res.reserve(total_steps % 100);

    int best_end_pos = -1;
    double best_end_dist = distance;
    double x = start.x;
    double y = start.y;
    double iL = 1 / (2 * _lr);

    for (int i = 0; i < total_steps; i++)
    {
        double steer = tan(steering_angle_deg);
        double beta = atan(steer / _lr);

        x += ds * cos(heading + beta);
        y += ds * sin(heading + beta);
        heading += (ds * cos(beta) * steer) * iL;

        double3 p;
        p.x = x;
        p.y = y;

        if (output_heading_in_degrees)
            p.z = to_degrees(heading);
        else
            p.z = heading;

        path_heading = compute_path_heading(p, end);
        steering_angle_deg = clip(path_heading - heading, -max_turning_angle, max_turning_angle);
        double dist = compute_euclidean_dist(p, end);

        convert_to_waypoint_coord(_center, _rate_w, _rate_h, p);

        if (p.x == last_x && p.y == last_y)
            continue;

        if (best_end_dist > dist)
        {
            best_end_dist = dist;
            best_end_pos = res.size();
        }

        last_x = static_cast<int>(p.x);
        last_y = static_cast<int>(p.y);

        res.push_back(p);
    }

    res.erase(res.begin() + best_end_pos + 1, res.end());
    return res;
}

bool ConstraintsCheckCPU::computeFeasibleForAngle(
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
    int upper_bound_ego_z)
{

    float c = cosf(angle_radians);
    float s = sinf(angle_radians);

    for (int i = -min_dist_z; i <= min_dist_z; i++)
        for (int j = -min_dist_x; j <= min_dist_x; j++)
        {
            int xl = round(j * c - i * s) + x;
            int zl = round(j * s + i * c) + z;

            if (xl < 0 || xl >= width)
                continue;

            if (zl < 0 || zl >= height)
                continue;

            if (xl >= lower_bound_ego_x && xl <= upper_bound_ego_x && zl >= upper_bound_ego_z && zl <= lower_bound_ego_z)
                continue;

            int segmentation_class = round(frame[zl * width + xl].x);

            if (classCost[segmentation_class] < 0)
            {
                // printf ("(%d, %d) not feasible on angle %f because of (%d, %d)\n", x, z, (180 * angle_radians) / 3.141592654F,  xl, zl);
                return false;
            }
        }

    return true;
}
