#include "../include/world_pose.h"
#include <cmath>

#define EARTH_RADIUS_meters 6378137

double WorldPose::distanceBetween(double lat1, double lon1, double lat2, double lon2)
{
    double dLat = (lat1 - lat2);
    double dLon = (lon1 - lon2);
    double a = 0.5 - cos(dLat) / 2 + cos(lat2) * cos(lat1) * (1 - cos(dLon)) / 2;
    return EARTH_RADIUS_meters * 2 * asin(sqrt(a));
} 
double WorldPose::distanceBetween(WorldPose &p1, WorldPose &p2)
{
    double lat1 = p1.lat().rad();
    double lat2 = p2.lat().rad();
    double lon1 = p1.lon().rad();
    double lon2 = p2.lon().rad();

    return distanceBetween(lat1, lon1, lat2, lon2);
}

double WorldPose::computeHeading(double lat1, double lon1, double lat2, double lon2)
{
    double y = sin(lon2 - lon1) * cos(lat2);
    double x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1);
    double theta = atan2(y, x);
    double p = (theta + DOUBLE_PI);
    while (p >= DOUBLE_PI)
        p -= DOUBLE_PI;
    return p;
}

angle WorldPose::computeHeading(WorldPose &p1, WorldPose &p2)
{
    double lat1 = p1.lat().rad();
    double lat2 = p2.lat().rad();
    double lon1 = p1.lon().rad();
    double lon2 = p2.lon().rad();

    return angle::rad(computeHeading(lat1, lon1, lat2, lon2));
}

extern "C"
{
    double world_pose_distance_between(double lat1, double lon1, double lat2, double lon2)
    {
        return WorldPose::distanceBetween(lat1, lon1, lat2, lon2);
    }

    double world_pose_compute_heading(double lat1, double lon1, double lat2, double lon2)
    {
        return WorldPose::computeHeading(lat1, lon1, lat2, lon2);
    }
}