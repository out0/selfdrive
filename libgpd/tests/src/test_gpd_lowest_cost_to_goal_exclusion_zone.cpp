#include <gtest/gtest.h>
#include <cmath>
#include <driveless/search_frame.h>
#include <driveless/waypoint.h>
#include "../../include/gpd.h"
#include "test_utils.h"

TEST(GPD_LowestCostToGoal, Test_InRange_NoObstacles_ExclusionZone)
{
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
    f1.processSafeDistanceZone({10, 10}, false);
    gpd.computeExclusionZone(f1, angle::rad(0.0));

    Waypoint p = gpd.findLowestCostWaypointToGoal(f1, {10, 10}, 50, 0, 0.0);
    ASSERT_EQ(p.x(), 50);
    ASSERT_EQ(p.z(), 1);
    ASSERT_EQ(p.heading(), angle::deg(0));
    //printf("%d, %d, %f\n", p.x(), p.z(), p.heading().deg());
    ASSERT_TRUE(f1.isTraversable(p.x(), p.z(), p.heading(), true));
}

TEST(GPD_LowestCostToGoal, Test_InRange_InvalidL1_Straight)
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
    f1.processSafeDistanceZone({10, 10}, false);
    gpd.computeExclusionZone(f1, angle::rad(0.0));
    Waypoint p = gpd.findLowestCostWaypointToGoal(f1, {10, 10}, 50, 0, 0.0);
    // ASSERT_EQ(p.x(), 50);
    // ASSERT_EQ(p.z(), 6);
    // ASSERT_EQ(p.heading(), angle::deg(0));
    //printf("%d, %d, %f\n", p.x(), p.z(), p.heading().deg());
    ASSERT_TRUE(f1.isTraversable(p.x(), p.z(), p.heading(), true));
    //
}

TEST(GPD_LowestCostToGoal, Test_InRange_InvalidL1_Curve)
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
    f1.processSafeDistanceZone({10, 10}, false);
    gpd.computeExclusionZone(f1, angle::deg(-90.0));
    Waypoint p = gpd.findLowestCostWaypointToGoal(f1, {10, 10}, 0, 21, angle::deg(-90).rad());
    ASSERT_TRUE(f1.isTraversable(p.x(), p.z(), p.heading(), true));
    //printf("%d, %d, %f\n", p.x(), p.z(), p.heading().deg());
        //dump_search_frame_debug(f1);

}
    