#include "../include/coord_conversion.h"
#include "../include/math_utils.h"
#include <cmath>

#define EARTH_RADIUS 6378137.0

CoordinateConverter::CoordinateConverter(WorldPose &origin,
                                         int width,
                                         int height,
                                         float perceptionWidthSize_m,
                                         float perceptionHeightSize_m) : _ogCenter(TO_INT(width / 2), TO_INT(width / 2), angle::rad(0)),
                                                                         _origin_compass_angle(origin.compass())
{
    _world_coord_scale = cos(origin.lat().rad());
    _rateW = width / perceptionWidthSize_m;
    _rateH = height / perceptionHeightSize_m;
    _invRateW = perceptionWidthSize_m / width;
    _invRateH = perceptionHeightSize_m / height;
}

double CoordinateConverter::__convert_map_heading_to_compass(double h)
{
    double a = h + _origin_compass_angle.rad();
    if (a > PI)
        a -= DOUBLE_PI;
    if (a < 0)
        return a + DOUBLE_PI;
    return a;
}
double CoordinateConverter::__convert_compass_to_map_heading(double hc)
{
    double a = hc - _origin_compass_angle.rad();
    if (a < 0) // compass should never be < 0
        a += DOUBLE_PI;
    if (a > PI)
        return a - DOUBLE_PI;
    return a;
}

MapPose CoordinateConverter::convert(WorldPose &world)
{
    double x = _world_coord_scale * EARTH_RADIUS * world.lon().rad();
    double y = -_world_coord_scale * EARTH_RADIUS * log(tan(QUARTER_PI + 0.5 * world.lat().rad()));
    double h = __convert_compass_to_map_heading(world.compass().rad());
    return MapPose(x, y, world.alt(), angle::rad(h));
}
WorldPose CoordinateConverter::convert(MapPose &pose)
{
    // double lat_rad = DOUBLE_PI * atan(exp(-pose.y() / (EARTH_RADIUS * _world_coord_scale))) / PI - HALF_PI;
    // double lon_rad = pose.x() / EARTH_RADIUS * _world_coord_scale;

    double lat = 2 * atan(exp(-pose.y() / (_world_coord_scale * EARTH_RADIUS))) - HALF_PI;
    double lon = pose.x() / (_world_coord_scale * EARTH_RADIUS);

    return WorldPose(
        angle::rad(lat),
        angle::rad(lon),
        pose.z(),
        angle::rad(__convert_map_heading_to_compass(pose.heading().rad())));
}

Waypoint CoordinateConverter::convert(MapPose &location, MapPose &target)
{
    double dx = target.x() - location.x();
    double dy = target.y() - location.y();
    double c = cos(-location.heading().rad());
    double s = sin(-location.heading().rad());

    double p0 = _rateH * (c * dx - s * dy);
    double p1 = _rateW * (s * dx + c * dy);

    int x = TO_INT(_ogCenter.x() + p1);
    int z = TO_INT(_ogCenter.z() - p0);

    return Waypoint(x, z, target.heading() - location.heading());
}
MapPose CoordinateConverter::convert(MapPose &location, Waypoint &target)
{
    double p0 = _ogCenter.z() - target.z();
    double p1 = target.x() - _ogCenter.x();
    double c = cos(location.heading().rad());
    double s = sin(location.heading().rad());

    double x = _invRateH * c * p0 - _invRateW * s * p1 + location.x();
    double y = _invRateH * s * p0 + _invRateW * c * p1 + location.y();

    return MapPose(
        x,
        y,
        location.z(),
        target.heading() + location.heading());
}
