#pragma once

#ifndef __WAYPOINT_DRIVELESS_H
#define __WAYPOINT_DRIVELESS_H

#include "angle.h"
#include "state.h"

// CODE:BEGIN

class Waypoint : public State
{
    bool __checked_as_feasible;
public:
    Waypoint(int x, int z, angle heading) : State(x, z, heading, 0.0) {}

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

    bool is_checked_as_feasible() { return this->__checked_as_feasible; }
    
    void set_checked_as_feasible(bool val) { 
//        printf ("checked as feasible: %d, %d\n", x(), z());
        this->__checked_as_feasible = val; 
    }
};

// CODE:END

#endif