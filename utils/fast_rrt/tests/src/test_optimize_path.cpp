#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <unordered_map>
#include "../../../cudac/include/cuda_frame.h"
#include <cmath>
#include "test_utils.h"
#include "../../include/graph.h"

#define PHYS_SIZE 34.641016151377535

TEST(TestOptimizeGraphs, TestOptimize_NoObstacle)
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
    // tstFrame.addArea(130, 90, 140, 70, 28);

    g.add(128, 128, angle::rad(0.0), -1, -1, 0.0);
    g.add(127, 121, angle::rad(-0.05952281504869461), 128, 128, 10.0);
    g.add(127, 116, angle::rad(-0.035784658044576645), 127, 121, 20.0);
    g.add(128, 105, angle::rad(0.010117189027369022), 127, 116, 30.0);
    g.add(105, 92, angle::rad(-0.1315600872039795), 128, 105, 40.0);
    g.add(96, 70, angle::rad(-0.353880912065506), 105, 92, 50.0);
    g.add(112, 37, angle::rad(-0.10171803086996078), 96, 70, 60.0);
    g.add(112, 17, angle::rad(-0.022149957716464996), 112, 37, 70.0);
    g.add(113, 7, angle::rad(0.015074538066983223), 112, 17, 80.0);
    g.add(114, 0, angle::rad(0.04908384382724762), 113, 7, 90.0);


    g.optimizeNode(ptr, 128, 105, 30.0, 1.0);

    exportGraph(&g, "test.png");
}