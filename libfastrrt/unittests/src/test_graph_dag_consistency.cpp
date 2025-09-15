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

TEST(TestGraphConsistencyCheck, CheckEmptyGraph)
{
    CudaGraph *graph = buildTestGraph();
    ASSERT_TRUE(graph->checkGraphIsDAG());
}

TEST(TestGraphConsistencyCheck, CheckOneNodeGraph)
{
    CudaGraph *graph = buildTestGraph();
    graph->add(128, 128, angle::rad(0), -1, -1, 0);
    ASSERT_TRUE(graph->checkGraphIsDAG());
}


TEST(TestGraphConsistencyCheck, CheckTwoNodesGraph)
{
    CudaGraph *graph = buildTestGraph();
    graph->add(128, 128, angle::rad(0), -1, -1, 0);
    graph->add(128, 78, angle::rad(0), 128, 128, 0);
    ASSERT_TRUE(graph->checkGraphIsDAG());
}


TEST(TestGraphConsistencyCheck, CheckCircle)
{
    CudaGraph *graph = buildTestGraph();
    //
    //  (78, 90) ---> (128, 78) ----> (128, 48)
    //   /\                               |
    //   |--------------------------------+
    graph->add(128, 128, angle::rad(0), -1, -1, 0);
    graph->add(128, 48, angle::rad(0), 78, 90, 0);
    graph->add(78, 90, angle::rad(0), 128, 78, 0);
    graph->add(128, 78, angle::rad(0), 128, 48, 0);   
    ASSERT_FALSE(graph->checkGraphIsDAG());
}

TEST(TestGraphConsistencyCheck, CheckTwoNodeCircle)
{
    CudaGraph *graph = buildTestGraph();
    //
    //  (78, 90) ---> (128, 78)
    //   /\             |
    //   |--------------+
    graph->add(128, 128, angle::rad(0), -1, -1, 0);
    graph->add(78, 90, angle::rad(0), 128, 78, 0);
    graph->add(128, 78, angle::rad(0), 78, 90, 0);   
    ASSERT_FALSE(graph->checkGraphIsDAG());
}


TEST(TestGraphConsistencyCheck, CheckPointSelf)
{
    CudaGraph *graph = buildTestGraph();
    //
    //  (78, 90) ---> (128, 78)
    //   /\             |
    //   |--------------+
    graph->add(128, 128, angle::rad(0), -1, -1, 0);
    graph->add(78, 90, angle::rad(0), 78, 90, 0);
    ASSERT_FALSE(graph->checkGraphIsDAG());
}