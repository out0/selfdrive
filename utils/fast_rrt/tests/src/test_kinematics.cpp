#include "../../../cudac/include/cuda_frame.h"
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include "test_utils.h"


extern double2 convert_waypoint_to_map_pose(int2 center, double inv_rate_w, double inv_rate_h, int2 coord);
extern int2 convert_map_pose_to_waypoint(int2 center, float rate_w, float rate_h, double2 coord);

#define PHYS_SIZE 34.641016151377535

TEST(TestKinematics, ConvertMapPoseWaypointOrigin)
{
    float rw = 256 / PHYS_SIZE;
    float rh = 256 / PHYS_SIZE;
    float inv_rw = PHYS_SIZE / 256;
    float inv_rh = PHYS_SIZE / 256;

    int2 center = {128, 128};
    int2 coord = {128, 128};

    double2 mapCoord = convert_waypoint_to_map_pose(center, inv_rw, inv_rh, coord);

    ASSERT_EQ(0, mapCoord.x);
    ASSERT_EQ(0, mapCoord.y);

    int2 waypoint = convert_map_pose_to_waypoint(center, rw, rh, mapCoord);

    ASSERT_EQ(128, waypoint.x);
    ASSERT_EQ(128, waypoint.y);
}

TEST(TestKinematics, ConvertMapPoseWaypointNonOrigin)
{
    float rw = 256 / PHYS_SIZE;
    float rh = 256 / PHYS_SIZE;
    float inv_rw = PHYS_SIZE / 256;
    float inv_rh = PHYS_SIZE / 256;

    int2 center = {128, 128};
    int2 coord = {108, 108};

    double2 mapCoord = convert_waypoint_to_map_pose(center, inv_rw, inv_rh, coord);

    int2 waypoint = convert_map_pose_to_waypoint(center, rw, rh, mapCoord);

    ASSERT_EQ(108, waypoint.x);
    ASSERT_EQ(108, waypoint.y);
}

TEST(TestKinematics, TestCheckKinematicPath)
{
    CudaGraph g(256, 256);
    float3 *ptr = createEmptySearchFrame(256, 256);
    angle maxSteering = angle::deg(40);
    int costs[] = {{0},
                   {1},
                   {2},
                   {3},
                   {4},
                   {5}};
    g.setPhysicalParams(PHYS_SIZE, PHYS_SIZE, maxSteering, 5.412658773);
    g.setClassCosts(costs, 6);
    g.setSearchParams({0, 0}, {-1, -1}, {-1, -1});


    g.add(128, 128, angle::rad(0), -1, -1, 0);

    int2 node = g.derivateNode(ptr, angle::rad(0), angle::deg(20), 50, 2, 128, 128);

    ASSERT_TRUE(g.checkFeasibleConnection(ptr, {128, 128}, node, 2, maxSteering));

    
}