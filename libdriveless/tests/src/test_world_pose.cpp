#include <gtest/gtest.h>
#include "../../include/world_pose.h"
#include "test_utils.h"
#include <cmath>

TEST(StateWorldPose, TestSetGet)
{
    WorldPose p1(angle::rad(1.2), angle::rad(-2.2), 32.1, angle::rad(0.34));

    ASSERT_TRUE(p1.lat() == angle::rad(1.2));
    ASSERT_TRUE(p1.lon() == angle::rad(-2.2));
    ASSERT_FLOAT_EQ(p1.alt(), 32.1);
    ASSERT_TRUE(p1.compass() == angle::rad(0.34));
}

TEST(StateWorldPose, TestOperators)
{
    WorldPose p1(angle::rad(1.2), angle::rad(-2.2), 32.1, angle::rad(0.34));
    WorldPose p2(angle::rad(1.2), angle::rad(-2.2), 32.1, angle::rad(0.34));
    WorldPose p3(angle::rad(2.2), angle::rad(-2.2), 32.1, angle::rad(0.34));

    ASSERT_TRUE(p1 == p2);
    ASSERT_TRUE(p1 != p3);
}

float compute_diff_percent(float d1, float d2)
{
    return 100 * (1 - d1 / d2);
}

TEST(StateWorldPose, TestDistanceBetween)
{
    WorldPose p1(angle::deg(5.566896), angle::deg(95.3672), 0, angle::rad(0));
    WorldPose p2(angle::deg(5.566607), angle::deg(95.370121), 0, angle::rad(0));
    double dist = WorldPose::distanceBetween(p1, p2);

    double dist_google = 324.93;
    ASSERT_TRUE(compute_diff_percent(dist, dist_google) < 0.5);

    WorldPose p3(angle::deg(5.566896), angle::deg(95.3672), 0, angle::rad(0));
    WorldPose p4(angle::deg(5.567333), angle::deg(95.367886), 0, angle::rad(0));
    dist_google = 89.86;
    dist = WorldPose::distanceBetween(p3, p4);
    ASSERT_TRUE(compute_diff_percent(dist, dist_google) < 0.5);

    WorldPose p5(angle::deg(-29.279371052090724), angle::deg(-56.91723210848266), 0, angle::rad(0));
    WorldPose p6(angle::deg(-27.92906183475516), angle::deg(-49.75414619790477), 0, angle::rad(0));
    dist = WorldPose::distanceBetween(p5, p6);
    dist_google = 713190;
    ASSERT_TRUE(compute_diff_percent(dist, dist_google) < 0.5);
}

TEST(StateWorldPose, TestComputeHeading)
{
    WorldPose p1(angle::deg(-29.279371052090724), angle::deg(-56.91723210848266), 0, angle::rad(0));
    WorldPose p2(angle::deg(-27.92906183475516), angle::deg(-49.75414619790477), 0, angle::rad(0));
    angle a = WorldPose::computeHeading(p1, p2);
    ASSERT_TRUE(a == angle::deg(79.61));
}
