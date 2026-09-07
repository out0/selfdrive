#pragma once

#ifndef __CUBIC_INTERPOLATOR_DRIVELESS_H
#define __CUBIC_INTERPOLATOR_DRIVELESS_H

#include <vector>
#include "waypoint.h"

class Interpolator
{
public:
    static std::vector<Waypoint> cubicSpline(std::vector<Waypoint> &dataPoints, int resolution = 10);
    static std::vector<Waypoint> hermite(int width, int height, Waypoint p1, Waypoint p2, float max_curvature);
    static std::vector<Waypoint> kinematicCurve(int width, int height, Waypoint start, angle steeringAngle, int wheelbase_px, int max_path_size);
};

#endif