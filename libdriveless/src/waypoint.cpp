#include "../include/waypoint.h"
#include "../include/math_utils.h"
#include <cmath>

/*Computes the euclidian distance between two waypoints */
double Waypoint::distanceBetween(Waypoint &p1, Waypoint &p2)
{
    return distanceBetween(p1.x(), p1.z(), p2.x(), p2.z());
}

/*Computes the euclidian distance between two waypoints */
double Waypoint::distanceBetween(int x1, int z1, int x2, int z2)
{
    double dz = z2 - z1;
    double dx = x2 - x1;
    return sqrt(dx * dx + dz * dz);
}

/*  Computes the heading between two waypoints relative to the BEV
    coordinate system (ref. docs/coordinate_systems.txt) */
double Waypoint::computeHeading(int x1, int z1, int x2, int z2)
{
    double dz = z2 - z1;
    double dx = x2 - x1;

    if (dx == 0 && dz == 0)
        return 0;

    double v1 = 0;
    if (dz != 0)
        v1 = atan2(-dz, dx);
    else
        v1 = atan2(0, dx);

    return HALF_PI - v1;
}
/*  Computes the heading between two waypoints relative to the BEV
    coordinate system (ref. docs/coordinate_systems.txt) */
angle Waypoint::computeHeading(Waypoint &p1, Waypoint &p2)
{
    return angle::rad(computeHeading(p1.x(), p1.z(), p2.x(), p2.z()));
}

double Waypoint::distanceToLine(Waypoint &line_p1, Waypoint &line_p2, Waypoint &p)
{
    return distanceToLine(line_p1.x(), line_p1.z(), line_p2.x(), line_p2.z(), p.x(), p.z());
}

double Waypoint::distanceToLine(int x1, int z1, int x2, int z2, int x_target, int z_target)
{
    double dz = z2 - z1;
    double dx = x2 - x1;

    if (dx == 0 && dz == 0)
        return 0;

    double num = dx * (z1 - z_target) - (x1 - x_target) * dz;
    double den = sqrt(dx * dx + dz * dz);
    return num / den;
}
Waypoint Waypoint::midPoint(Waypoint &p1, Waypoint &p2)
{
    return Waypoint(
        TO_INT((p2.x() + p1.x()) / 2),
        TO_INT((p2.z() + p1.z()) / 2),
        p1.heading());
}

Waypoint Waypoint::clip(Waypoint p, int width, int height)
{
    int x = p.x();
    int z = p.z();

    if (x < 0)
        x = 0;
    else if (x >= width)
        x = width - 1;
    if (z < 0)
        z = 0;
    else if (z >= height)
        z = height - 1;
    return Waypoint(x, z, p.heading());
}

Waypoint Waypoint::clone()
{
    return Waypoint(x(), z(), heading());
}

Waypoint *Waypoint::cloneToPtr()
{
    return new Waypoint(x(), z(), heading());
}

// multilanguage bind

extern "C"
{
    double waypoint_distance_between(int x1, int z1, int x2, int z2)
    {
        return Waypoint::distanceBetween(x1, z1, x2, z2);
    }

    /*  Computes the heading between two waypoints relative to the BEV
        coordinate system (ref. docs/coordinate_systems.txt) */
    double waypoint_compute_heading(int x1, int z1, int x2, int z2)
    {
        return Waypoint::computeHeading(x1, z1, x2, z2);
        
    }

    double waypoint_distance_to_line(int x1, int z1, int x2, int z2, int x_target, int z_target)
    {
        return Waypoint::distanceToLine(x1, z1, x2, z2, x_target, z_target);
    }
}