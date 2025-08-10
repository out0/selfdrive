#pragma once

#ifndef __STATE_WAYPOINT_DRIVELESS_H
#define __STATE_WAYPOINT_DRIVELESS_H

#include "angle.h"

// CODE:BEGIN

class Waypoint
{
    int _x;
    int _z;
    angle _heading;

private:

public:
    Waypoint(int x, int z, angle heading) : _x(x), _z(z), _heading(heading) {}

    constexpr inline int x() { return _x; }
    constexpr inline int z() { return _z; }
    inline angle heading() { return _heading; }

    inline bool operator==(const Waypoint &other)
    {
        return _x == other._x &&
               _z == other._z &&
               _heading == other._heading;
    }
    inline bool operator!=(const Waypoint &other)
    {
        return !(*this == other);
    }

    /*Computes the euclidian distance between two waypoints */
    static double distanceBetween(Waypoint &p1, Waypoint &p2);

    static inline double computeHeading(int x1, int z1, int x2, int z2);
    static inline double distanceBetween(int x1, int z1, int x2, int z2);
    static double distanceToLine(int x1, int z1, int x2, int z2, int x_target, int z_target);


    /*  Computes the heading between two waypoints relative to the BEV
        coordinate system (ref. docs/coordinate_systems.txt) */
    static angle computeHeading(Waypoint &p1, Waypoint &p2);

    static double distanceToLine(Waypoint &line_p1, Waypoint &line_p2, Waypoint &p);

    static Waypoint midPoint(Waypoint &p1, Waypoint &p2);

    static Waypoint clip(Waypoint p, int width, int height);

    Waypoint clone();

    Waypoint * cloneToPtr();
};

// CODE:END

#endif