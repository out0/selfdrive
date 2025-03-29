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

    g.add(128, 230, angle::rad(0.0), -1, -1, 0);
    int2 n1 = g.derivateNode(ptr, angle::rad(0), angle::deg(-30), 100, 1, 128, 230);
    ASSERT_NE(-1, n1.x);
    g.acceptDerivatedNodes();

    exportGraph(&g, "test.png");

    int2 n2 = g.derivateNode(ptr, angle::rad(0), angle::deg(30), 100, 1, n1.x, n1.y);
    ASSERT_NE(-1, n2.x);
    // ASSERT_TRUE(g->connectToGraph(og->getFramePtr(), 128, 230, 100, 200));

    g.acceptDerivatedNodes();

    exportGraph(&g, "test.png");

    int2 n3 = g.derivateNode(ptr, angle::rad(0), angle::deg(35), 50, 1, n2.x, n2.y);
    ASSERT_NE(-1, n3.x);

    g.acceptDerivatedNodes();

    exportGraph(&g, "test.png");

    int2 n4a = g.derivateNode(ptr, angle::rad(0), angle::deg(35), 50, 1, n3.x, n3.y);
    ASSERT_NE(-1, n4a.x);

    g.acceptDerivatedNodes();

    int2 parent = g.getParent(n4a.x, n4a.y);
    ASSERT_EQ(parent.x, n3.x);
    ASSERT_EQ(parent.y, n3.y);

    exportGraph(&g, "test.png");

    int2 n4b = g.derivateNode(ptr, angle::rad(0), angle::deg(-15), 50, 1, n3.x, n3.y);
    ASSERT_NE(-1, n4b.x);

    g.acceptDerivatedNodes();

    exportGraph(&g, "test.png");

    ASSERT_TRUE(g.checkFeasibleConnection(ptr, n4b, {118, 0}, 1, maxSteering));

    // I NEED TO COMPUTE PATH HEADING...

    g.add(118, 0, angle::rad(0.0), -1, -1, 0);

    int2 n5 = g.derivateNode(ptr, angle::deg(0), angle::deg(0), 190, 1, 128, 230);
    ASSERT_NE(-1, n5.x);

    // node n4a points to node n3 before optimization
    parent = g.getParent(n4a.x, n4a.y);
    ASSERT_EQ(parent.x, n3.x);
    ASSERT_EQ(parent.y, n3.y);

    g.optimizeGraph(ptr, angle::rad(0), 40.0, 1);

    g.acceptDerivatedNodes();

    // since nn5 is close to n4b and has a lower cost because it
    // has a better heading and comes straight from 128,128,
    // optimizeGraphWithNode should make n4b point to n5
    parent = g.getParent(n4a.x, n4a.y);
    ASSERT_EQ(parent.x, n5.x);
    ASSERT_EQ(parent.y, n5.y);

    // ASSERT_NE(-1, n3.x);

    // g->add(128, 108, 0.0, 128, 230, 0);
    // g->add(128, 88, 0.0, 128, 108, 0);
    // g->add(128, 48, 0.0, 128, 88, 0);
    // g->add(150, 0, 0.0, 128, 48, 0);
}