#include <gtest/gtest.h>
#include <cmath>
#include <driveless/search_frame.h>
#include <driveless/waypoint.h>
#include "../../include/gpd.h"

TEST(GPD_WithHeading, Test_InRange_NoObstacles)
{
    return;
    GoalPointDiscover gpd;

    SearchFrame f1(100, 100, {5, 5}, {15, 15});

    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
        {
            f1[{j, i}].x = 1;
        }

    f1.setClassColors({{255, 255, 255},
                       {0, 0, 0}});

    f1.setClassCosts({-1, 0});

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            for (int h = -90; h <= 90; h += 45)
            {
                auto heading = angle::deg(h);
                Waypoint p = gpd.findLowestCostWaypointWithHeading(f1, {10, 10}, x, z, heading.rad());
                if (p.x() != x || p.z() != z || p.heading() != heading)
                {
                    printf("expected (%d, %d, %f deg), obtained (%d, %d, %f deg)\n", x, z, heading.deg(), p.x(), p.z(), p.heading().deg());
                    FAIL();
                }
            }
}

TEST(GPD_WithHeading, Test_InRange_InvalidL1_Straight)
{
    GoalPointDiscover gpd;

    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
        {
            f1[{j, i}].x = 1;
        }

    f1.setClassColors({{255, 255, 255},
                       {0, 0, 0}});

    f1.setClassCosts({-1, 0});

    // setting the L1 position as obstacle
    f1[{50, 0}].x = 0;

    /// in this test, all of the points are feasible, but L1.
    // Thus, the GPD must find a L1 neighbor node


    Waypoint p = gpd.findLowestCostWaypointWithHeading(f1, {10, 10}, 50, 0, 0.0);
    ASSERT_TRUE(f1.isTraversable(p.x(), p.z(), p.heading(), true));
}

TEST(GPD_WithHeading, Test_InRange_InvalidL1_Curve)
{
    GoalPointDiscover gpd;

    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
        {
            f1[{j, i}].x = 1;
        }

    f1.setClassColors({{255, 255, 255},
                       {0, 0, 0}});

    f1.setClassCosts({-1, 0});

    // setting the L1 position as obstacle
    for (int x = 0; x < 30; x++)
        for (int z = 0; z < 20; z++)
            f1[{x, z}].x = 0;

    /// in this test, all of the points are feasible, but L1.
    // Thus, the GPD must find a L1 neighbor node

    Waypoint p = gpd.findLowestCostWaypointWithHeading(f1, {10, 10}, 0, 21, angle::deg(-45).rad());
    ASSERT_TRUE(f1.isTraversable(p.x(), p.z(), p.heading(), true));

}

TEST(GPD_WithHeading, Test_Nearby_NoObstacles)
{
    return;
    GoalPointDiscover gpd;

    SearchFrame f1(100, 100, {5, 5}, {15, 15});

    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
        {
            f1[{j, i}].x = 1;
        }

    f1.setClassColors({{255, 255, 255},
                       {0, 0, 0}});

    f1.setClassCosts({-1, 0});

    auto heading = angle::deg(22.5);
    Waypoint p = gpd.findLowestCostWaypointWithHeading(f1, {10, 10}, 110, -12, heading.rad());
    ASSERT_EQ(p.z(), 0);
    ASSERT_EQ(p.x(), 110);
    ASSERT_EQ(p.heading(), heading);
}

TEST(GPD_WithHeading, Test_Nearby_InvalidL1_Straight)
{
    GoalPointDiscover gpd;

    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
        {
            f1[{j, i}].x = 1;
        }

    f1.setClassColors({{255, 255, 255},
                       {0, 0, 0}});

    f1.setClassCosts({-1, 0});

    // setting the L1 position as obstacle
    f1[{50, 0}].x = 0;

    /// in this test, all of the points are feasible, but L1.
    // Thus, the GPD must find a L1 neighbor node


    Waypoint p = gpd.findLowestCostWaypointWithHeading(f1, {10, 10}, 50, -10, 0.0);
    ASSERT_TRUE(f1.isTraversable(p.x(), p.z(), p.heading(), true));
}

TEST(GPD_WithHeading, Test_Nearby_InvalidL1_Curve)
{
    GoalPointDiscover gpd;

    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
        {
            f1[{j, i}].x = 1;
        }

    f1.setClassColors({{255, 255, 255},
                       {0, 0, 0}});

    f1.setClassCosts({-1, 0});

    // setting the L1 position as obstacle
    for (int x = 0; x < 30; x++)
        for (int z = 0; z < 20; z++)
            f1[{x, z}].x = 0;

    /// in this test, all of the points are feasible, but L1.
    // Thus, the GPD must find a L1 neighbor node

    Waypoint p = gpd.findLowestCostWaypointWithHeading(f1, {10, 10}, -10, 21, angle::deg(-45).rad());
    ASSERT_TRUE(f1.isTraversable(p.x(), p.z(), p.heading(), true));

}

TEST(GPD_WithHeading, Test_Far_NoObstacles)
{
    return;
    GoalPointDiscover gpd;

    SearchFrame f1(100, 100, {5, 5}, {15, 15});

    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
        {
            f1[{j, i}].x = 1;
        }

    f1.setClassColors({{255, 255, 255},
                       {0, 0, 0}});

    f1.setClassCosts({-1, 0});

    auto heading = angle::deg(22.5);
    Waypoint p = gpd.findLowestCostWaypointWithHeading(f1, {10, 10}, 110, -120, heading.rad());
    ASSERT_EQ(p.z(), 0);
    ASSERT_EQ(p.x(), 110);
    ASSERT_EQ(p.heading(), heading);
}

TEST(GPD_WithHeading, Test_Far_InvalidL1_Straight)
{
    GoalPointDiscover gpd;

    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
        {
            f1[{j, i}].x = 1;
        }

    f1.setClassColors({{255, 255, 255},
                       {0, 0, 0}});

    f1.setClassCosts({-1, 0});

    // setting the L1 position as obstacle
    f1[{50, 0}].x = 0;

    /// in this test, all of the points are feasible, but L1.
    // Thus, the GPD must find a L1 neighbor node


    Waypoint p = gpd.findLowestCostWaypointWithHeading(f1, {10, 10}, 50, -100, 0.0);
    ASSERT_TRUE(f1.isTraversable(p.x(), p.z(), p.heading(), true));
}

TEST(GPD_WithHeading, Test_Far_InvalidL1_Curve)
{
    GoalPointDiscover gpd;

    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
        {
            f1[{j, i}].x = 1;
        }

    f1.setClassColors({{255, 255, 255},
                       {0, 0, 0}});

    f1.setClassCosts({-1, 0});

    // setting the L1 position as obstacle
    for (int x = 0; x < 30; x++)
        for (int z = 0; z < 20; z++)
            f1[{x, z}].x = 0;

    /// in this test, all of the points are feasible, but L1.
    // Thus, the GPD must find a L1 neighbor node

    Waypoint p = gpd.findLowestCostWaypointWithHeading(f1, {10, 10}, -100, 21, angle::deg(-45).rad());
    ASSERT_TRUE(f1.isTraversable(p.x(), p.z(), p.heading(), true));

}
