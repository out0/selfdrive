#pragma once

#ifndef __COORD_CONVERSION_DRIVELESS_H
#define __COORD_CONVERSION_DRIVELESS_H

#include "waypoint.h"
#include "world_pose.h"
#include "map_pose.h"
#include <cmath>


class CoordinateConverter
{
private:
    float _world_coord_scale;
    angle _origin_compass_angle;
    float _rateW;
    float _rateH;
    float _invRateW;
    float _invRateH;
    Waypoint _ogCenter;

    double __convert_map_heading_to_compass(double h);
    double __convert_compass_to_map_heading(double hc);



public:
    CoordinateConverter(WorldPose &origin, int width, int height, float perceptionWidthSize_m, float perceptionHeightSize_m);

    MapPose convert (WorldPose &pose);
    WorldPose convert (MapPose &pose);

    Waypoint convert (MapPose &location, MapPose &target);
    MapPose convert (MapPose &location, Waypoint &pose);
};

#endif
