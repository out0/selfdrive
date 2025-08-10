#include <gtest/gtest.h>
#include "../../include/coord_conversion.h"
#include "test_utils.h"
#include <cmath>

#define TST_COORD_WIDTH 600
#define TST_COORD_HEIGHT 600
#define TST_COORD_REAL_W 60
#define TST_COORD_REAL_H 60

CoordinateConverter getConverter()
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

void execTest(CoordinateConverter &conv, MapPose &location, double x, double y, int expected_x, int expected_z)
{
    MapPose mapPose(x, y, 0, angle::rad(0));
    Waypoint wp = conv.convert(location, mapPose);
    ASSERT_EQ(wp.x(), expected_x);
    ASSERT_EQ(wp.z(), expected_z);
}

TEST(CoordConversionMapPose, TestHeading0_0deg)
{
    CoordinateConverter conv = getConverter();
    MapPose location(0, 0, 0, angle::rad(0));
    execTest(conv, location, 30, 0, 300, 0);
    execTest(conv, location, -30, 0, 300, 600);
    execTest(conv, location, 0, 30, 600, 300);
    execTest(conv, location, 0, -30, 0, 300);
    execTest(conv, location, 30, 30, 600, 0);
    execTest(conv, location, -30, 30, 600, 600);
    execTest(conv, location, 30, -30, 0, 0);
    execTest(conv, location, -30, -30, 0, 600);
}

TEST(CoordConversionMapPose, TestHeading0_45deg)
{
    CoordinateConverter conv = getConverter();
    MapPose location(0, 0, 0, angle::rad(0));
    execTest(conv, location, 30, 30, 600, 0);
    execTest(conv, location, -30, 30, 600, 600);
    execTest(conv, location, 30, -30, 0, 0);
    execTest(conv, location, -30, -30, 0, 600);
    execTest(conv, location, 30, 30, 600, 0);
}

TEST(CoordConversionMapPose, TestHeading45_45deg)
{
    CoordinateConverter conv = getConverter();
    double rt = sqrt(2);
    MapPose location(0, 0, 0, angle::deg(45));
    execTest(conv, location, 15*rt, 15*rt, 300, 0);
    execTest(conv, location, -15*rt, 15*rt, 600, 300);
    execTest(conv, location, 15*rt, -15*rt, 0, 300);
    execTest(conv, location, -15*rt, -15*rt, 300, 600);
}

TEST(CoordConversionMapPose, TestHeadingMinus45_45deg)
{
    CoordinateConverter conv = getConverter();
    double rt = sqrt(2);
    MapPose location(0, 0, 0, angle::deg(-45));
    execTest(conv, location, 15*rt, 15*rt, 600, 300);
    execTest(conv, location, -15*rt, 15*rt, 300, 600);
    execTest(conv, location, 15*rt, -15*rt, 300, 0);
    execTest(conv, location, -15*rt, -15*rt, 0, 300);
}