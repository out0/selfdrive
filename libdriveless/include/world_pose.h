#pragma once

#ifndef __STATE_WORLD_POSE_DRIVELESS_H
#define __STATE_WORLD_POSE_DRIVELESS_H

#include "angle.h"

// CODE:BEGIN

class WorldPose
{
    angle _lat;
    angle _lon;
    double _alt;
    angle _compass;

private:
public:
    WorldPose(angle lat, angle lon, double alt, angle compass) : _lat(lat), _lon(lon), _alt(alt), _compass(compass) {}

    angle lat()
    {
        return _lat;
    }

    angle lon()
    {
        return _lon;
    }

    double alt()
    {
        return _alt;
    }

    angle compass()
    {
        return _compass;
    }

    inline bool operator==(const WorldPose &other)
    {
        return __TOLERANCE_EQUALITY(_alt, other._alt) &&
               _lat == other._lat &&
               _lon == other._lon &&
               _compass == other._compass;
    }
    inline bool operator!=(const WorldPose &other)
    {
        return !(*this == other);
    }

    /*Compute the Haversine distance between two world absolute poses (ignoring heading)
        Args:
            p1 (WorldPose): origin (lat, lon)
            p2 (WorldPose): dest (lat, lon)

        Returns:
            float: distance in meters
    */
    static double distanceBetween(WorldPose &p1, WorldPose &p2);
    /* Computes the bearing (World heading or forward Azimuth) for two world poses

        Args:
            p1 (WorldPose): origin (lat, lon)
            p2 (WorldPose): dest (lat, lon)

        Returns:
            angle in radians
    */
    static angle computeHeading(WorldPose &p1, WorldPose &p2);

    static inline double distanceBetween(double lat1, double lon1, double lat2, double lon2);
    
    static inline double computeHeading(double lat1, double lon1, double lat2, double lon2);
};

// CODE:END

#endif