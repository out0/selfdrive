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
}