#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <driveless/coord_conversion.h>
#include <driveless/world_pose.h>
#include <driveless/cuda_ptr.h>
#include "test_utils.h"

#define PHYS_SIZE 34.641016151377535

CudaGraph *buildTestGraph()
{
    CudaGraph *g = new CudaGraph(256, 256);
    angle maxSteering = angle::deg(40);
    std::vector<float> costs = {
        {0},
        {1},
        {2},
        {3},
        {4},
        {5}};

    g->setPhysicalParams(256, 256, maxSteering, 5.412658773);
    g->setClassCosts(costs);
    g->setSearchParams({0, 0}, {-1, -1}, {-1, -1});
    return g;
}

TEST(TestGraphSolveCollisions, SingleCollisionSolve)
{
    CudaGraph *graph = buildTestGraph();
    angle p = angle::rad(0);

    graph->addStart(128, 128, p);                 // A
    /**/ graph->add(100, 100, p, 128, 128, 0);    // B
    /**/ graph->add(135, 100, p, 128, 128, 0);    // C
    /******/ graph->add(110, 70, p, 135, 100, 0); // D
    /******/ graph->add(145, 70, p, 135, 100, 0); // E

    ASSERT_EQ(graph->getChildCount(128, 128), 2);
    ASSERT_EQ(graph->getChildCount(100, 100), 0);
    ASSERT_EQ(graph->getChildCount(135, 100), 2);

    // TODO: fix this test. the collision operation is change to GRAPH_TYPE_COLLISION AND decrement node count.
    //  the test must reflect this...

    // COLLISION in B --> D
    graph->setCollision(110, 70, 100, 100, p, 0);

    graph->solveCollisions();

    ASSERT_EQ(graph->getChildCount(128, 128), 2);
    ASSERT_EQ(graph->getChildCount(100, 100), 1);
    ASSERT_EQ(graph->getChildCount(135, 100), 1);

    // This collision solve should only change the type of
    ASSERT_TRUE(graph->checkGraphIsDAG());

    assertInt2Equal(graph->getParent(128, 128), {-1, -1});
    assertInt2Equal(graph->getParent(100, 100), {128, 128});
    assertInt2Equal(graph->getParent(135, 100), {128, 128});
    assertInt2Equal(graph->getParent(110, 70), {100, 100});
    assertInt2Equal(graph->getParent(145, 70), {135, 100});

    ASSERT_EQ(graph->getType(110, 70), GRAPH_TYPE_NODE);
}

TEST(TestGraphSolveCollisions, CollisionSolveGraphErase)
{
    CudaGraph *graph = buildTestGraph();
    angle p = angle::rad(0);

    graph->add(128, 128, p, -1, -1, 0);             // A
    /**/ graph->add(100, 100, p, 128, 128, 0);      // B
    /**/ graph->add(135, 100, p, 128, 128, 0);      // C
    /******/ graph->add(110, 70, p, 135, 100, 0);   // D
    /**********/ graph->add(90, 50, p, 110, 70, 0); // D_A
    /**********/ graph->add(40, 35, p, 110, 70, 0); // D_B
    /******/ graph->add(145, 70, p, 135, 100, 0);   // E

    ASSERT_EQ(graph->getChildCount(128, 128), 2);
    ASSERT_EQ(graph->getChildCount(100, 100), 0);
    ASSERT_EQ(graph->getChildCount(135, 100), 2);
    ASSERT_EQ(graph->getChildCount(110, 70), 2);
    ASSERT_EQ(graph->getChildCount(90, 50), 0);
    ASSERT_EQ(graph->getChildCount(40, 35), 0);
    ASSERT_EQ(graph->getChildCount(145, 70), 0);

    // COLLISION in B --> D
    graph->setCollision(110, 70, 100, 100, p, 0);
    graph->solveCollisions();

    // This collision solve should only change the type of
    ASSERT_TRUE(graph->checkGraphIsDAG());

    assertInt2Equal(graph->getParent(128, 128), {-1, -1});
    assertInt2Equal(graph->getParent(100, 100), {128, 128});
    assertInt2Equal(graph->getParent(135, 100), {128, 128});
    assertInt2Equal(graph->getParent(110, 70), {100, 100});
    assertInt2Equal(graph->getParent(145, 70), {135, 100});

    ASSERT_EQ(graph->getType(90, 50), GRAPH_TYPE_NULL);
    ASSERT_EQ(graph->getType(40, 35), GRAPH_TYPE_NULL);
    ASSERT_EQ(graph->getType(110, 70), GRAPH_TYPE_NODE);

    ASSERT_EQ(graph->getChildCount(128, 128), 2);
    ASSERT_EQ(graph->getChildCount(100, 100), 1);
    ASSERT_EQ(graph->getChildCount(135, 100), 1);
    ASSERT_EQ(graph->getChildCount(110, 70), 0);
    ASSERT_EQ(graph->getChildCount(145, 70), 0);
}

TEST(TestGraphSolveCollisions, DualCollision)
{
    CudaGraph *graph = buildTestGraph();
    angle p = angle::rad(0);

    graph->add(128, 128, p, -1, -1, 0);             // A
    /**/ graph->add(100, 100, p, 128, 128, 0);      // B
    /**/ graph->add(135, 100, p, 128, 128, 0);      // C
    /******/ graph->add(110, 70, p, 135, 100, 0);   // D
    /**********/ graph->add(90, 50, p, 110, 70, 0); // D_A
    /**********/ graph->add(40, 35, p, 110, 70, 0); // D_B
    /******/ graph->add(145, 70, p, 135, 100, 0);   // E
    /******/ graph->add(185, 60, p, 135, 100, 0);   // F

    // COLLISION in B --> D
    graph->setCollision(110, 70, 100, 100, p, 0);
    // COLLISION in F --> D_A
    graph->setCollision(90, 50, 185, 60, p, 0);

    graph->solveCollisions();

    // This collision solve should only change the type of
    ASSERT_TRUE(graph->checkGraphIsDAG());

    assertInt2Equal(graph->getParent(128, 128), {-1, -1});
    assertInt2Equal(graph->getParent(100, 100), {128, 128});
    assertInt2Equal(graph->getParent(135, 100), {128, 128});
    assertInt2Equal(graph->getParent(110, 70), {100, 100});
    assertInt2Equal(graph->getParent(145, 70), {135, 100});

    ASSERT_EQ(graph->getType(90, 50), GRAPH_TYPE_NODE); // MAINTAIN D_A
    assertInt2Equal(graph->getParent(90, 50), {185, 60});
    ASSERT_EQ(graph->getType(40, 35), GRAPH_TYPE_NULL);

    ASSERT_EQ(graph->getType(110, 70), GRAPH_TYPE_NODE);
    assertInt2Equal(graph->getParent(110, 70), {100, 100});
}