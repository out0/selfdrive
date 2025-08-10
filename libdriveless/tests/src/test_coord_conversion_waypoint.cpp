#include <gtest/gtest.h>
#include "../../include/coord_conversion.h"
#include "test_utils.h"
#include <cmath>

#define TST_COORD_WIDTH 600
#define TST_COORD_HEIGHT 600
#define TST_COORD_REAL_W 60
#define TST_COORD_REAL_H 60

CoordinateConverter getWpConverter()
{
    WorldPose origin(angle::rad(0),
                     angle::rad(0),
                     0,
                     angle::rad(0));

    CoordinateConverter conv(origin,
                             TST_COORD_WIDTH,
                             TST_COORD_HEIGHT,
                             TST_COORD_REAL_W,
                             TST_COORD_REAL_H);
    return conv;
}

void execTest(CoordinateConverter &conv, MapPose &location, int x, int z, double expected_x, double expected_y)
{
    Waypoint wp(x, z, angle::rad(0));
    MapPose pose = conv.convert(location, wp);
    ASSERT_FLOAT_EQ(pose.x(), expected_x);
    ASSERT_FLOAT_EQ(pose.y(), expected_y);
}

TEST(CoordConversionWaypoint, TestHeading0_0deg)
{
    CoordinateConverter conv = getWpConverter();
    MapPose location(0, 0, 0, angle::rad(0));
    execTest(conv, location, 300, 0, 30, 0);
    execTest(conv, location, 300, 600, -30, 0);
    execTest(conv, location, 600, 300, 0, 30);
    execTest(conv, location, 0, 300, 0, -30);
}

TEST(CoordConversionWaypoint, TestHeading0_45deg)
{
    CoordinateConverter conv = getWpConverter();
    MapPose location(0, 0, 0, angle::rad(0));
    execTest(conv, location, 600, 0, 30, 30);
    execTest(conv, location, 600, 600, -30, 30);
    execTest(conv, location, 0, 0, 30, -30);
    execTest(conv, location, 0, 600, -30, -30);
}

TEST(CoordConversionWaypoint, TestHeading45_45deg)
{
    double rt = sqrt(2);
    CoordinateConverter conv = getWpConverter();
    MapPose location(0, 0, 0, angle::deg(45));
    execTest(conv, location, 300, 0, 15 * rt, 15 * rt);
    execTest(conv, location, 600, 300, -15 * rt, 15 * rt);
    execTest(conv, location, 0, 300, 15 * rt, -15 * rt);
    execTest(conv, location, 300, 600, -15 * rt, -15 * rt);
}

TEST(CoordConversionWaypoint, TestHeadingMinus45_45deg)
{
    return;
    double rt = sqrt(2);
    CoordinateConverter conv = getWpConverter();
    MapPose location(0, 0, 0, angle::deg(-45));
    execTest(conv, location, 600, 300, 15 * rt, 15 * rt);
    execTest(conv, location, 300, 600, -15 * rt, 15 * rt);
    execTest(conv, location, 300, 0, 15 * rt, -15 * rt);
    execTest(conv, location, 0, 300, -15 * rt, -15 * rt);
}