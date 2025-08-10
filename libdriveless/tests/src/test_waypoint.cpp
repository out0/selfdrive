#include <gtest/gtest.h>
#include "../../include/waypoint.h"
#include "test_utils.h"
#include <cmath>

TEST(StateWaypoint, TestSetGet)
{
    Waypoint p(16, 10, angle::deg(60));
    ASSERT_EQ(16, p.x());
    ASSERT_EQ(10, p.z());
    ASSERT_DEQ(60, p.heading().deg());
}


TEST(StateWaypoint, TestOperators)
{
    Waypoint p1(16, 10, angle::deg(60));
    Waypoint p2(16, 10, angle::deg(60));
    Waypoint p3(17, 10, angle::deg(60));
    Waypoint p4(16, 11, angle::deg(60));
    Waypoint p5(16, 11, angle::deg(61));

    ASSERT_TRUE(p1 == p2);
    ASSERT_FALSE(p1 != p2);
    ASSERT_TRUE(p1 != p3);
    ASSERT_TRUE(p1 != p4);
    ASSERT_TRUE(p1 != p5);
    
}


TEST(StateWaypoint, TestComputeDistance)
{
    Waypoint p1(0, 0, angle::deg(0));
    Waypoint p2(10, 10, angle::deg(0));
    Waypoint p3(-10, -10, angle::deg(0));
    ASSERT_DEQ(Waypoint::distanceBetween(p1, p2), 10 * sqrt(2));
    ASSERT_DEQ(Waypoint::distanceBetween(p2, p2), 0);
    ASSERT_DEQ(Waypoint::distanceBetween(p3, p2), 20 * sqrt(2)); 
}

TEST(StateWaypoint, TestComputeHeading)
{
    Waypoint p1(0, 0, angle::deg(0));
    Waypoint p2(1, 1, angle::deg(0));
    Waypoint p3(2, 0, angle::deg(0));
    Waypoint p4(2, 1, angle::deg(0));
    Waypoint p5(0, 1, angle::deg(0));
    Waypoint p6(2, 2, angle::deg(0));
    Waypoint p7(0, 2, angle::deg(0));
    Waypoint p8(0, 4, angle::deg(0));


    ASSERT_TRUE(Waypoint::computeHeading(p1, p1) == angle::deg(0));
    ASSERT_TRUE(Waypoint::computeHeading(p2, p1) == angle::deg(-45));
    ASSERT_TRUE(Waypoint::computeHeading(p2, p3) == angle::deg(45));
    ASSERT_TRUE(Waypoint::computeHeading(p2, p4) == angle::deg(90));
    ASSERT_TRUE(Waypoint::computeHeading(p2, p5) == angle::deg(-90));
    ASSERT_TRUE(Waypoint::computeHeading(p2, p6) == angle::deg(135));
    ASSERT_TRUE(Waypoint::computeHeading(p2, p7) == angle::deg(225));
    ASSERT_TRUE(Waypoint::computeHeading(p1, p8) == angle::deg(180));
}

