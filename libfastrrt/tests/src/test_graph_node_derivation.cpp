#include "../../../cudac/include/cuda_frame.h"
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include "test_utils.h"

#define PHYS_SIZE 34.641016151377535

TEST(TestGraph, TestDrawPathCPU)
{
    int c = 256 * 256 * 3;
    float *frame = new float[c];
    for (int i = 0; i < c; i++)
        frame[i] == 3;

    CudaGraph g(256, 256);

    g.setPhysicalParams(PHYS_SIZE, PHYS_SIZE, angle::deg(40), 5.412658773);
    g.add(128, 128, angle::rad(0.0), -1, -1, 0);

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

    int2 lastNode = g.derivateNode(ptr, angle::rad(0), angle::deg(20), 70, 1, 128, 128);
    g.acceptDerivedNode({128, 128}, lastNode);
    // exportGraph(&g, "test.png");

    ASSERT_NE(lastNode.x, -1);
    ASSERT_NE(lastNode.y, -1);

    auto listNodes = g.list();

    ASSERT_EQ(2, listNodes.size());
}

TEST(TestGraph, TestDrawCurveOnEdge)
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

    g.add(128, 0, angle::rad(0.0), -1, -1, 0);
    g.derivateNode(ptr, angle::rad(0), angle::rad(15), 100, 1, 128, 0);
    g.acceptDerivedNodes();
    int2 parentOrig = g.getParent(128, 0);

    ASSERT_EQ(g.getType(128, 0), GRAPH_TYPE_NODE);
    ASSERT_EQ(parentOrig.x, -1);
    ASSERT_EQ(parentOrig.y, -1);
}

int countTempNodes(CudaGraph *g)
{
    int4 *ptr = g->getFramePtr()->getCudaPtr();
    int count_temp = 0;
    for (int i = 0; i < g->height(); i++)
        for (int j = 0; j < g->width(); j++)
            if (ptr[i * g->width() + j].z == GRAPH_TYPE_TEMP)
                count_temp++;
    return count_temp;
}

TEST(TestGraph, TestDrawCurve)
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
    g.add(128, 128, angle::rad(0.0), -1, -1, 0);

    g.derivateNode(ptr, angle::rad(0), angle::rad(0), 100, 1, 128, 128);
    g.acceptDerivedNodes();
    ASSERT_EQ(2, g.list().size());
}

TEST(TestGraph, TestDrawManyPathsGPU)
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
    g.add(128, 128, angle::rad(0.0), -1, -1, 0);

    for (int i = 0; i < 10; i++)
    {
        g.derivateNode(ptr, angle::rad(0), angle::rad(0), (float)100.0, (float)1.0, 128, 128);
        g.acceptDerivedNodes();
    }

    ASSERT_GE(g.list().size(), 10);
    exportGraph(&g, "test3.png");
}


TEST(TestGraph, TestBugDeriveNodesEraseParents)
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
    g.add(128, 128, angle::rad(0.0), -1, -1, 0);
    
    int type = (*g.getFramePtr())[{128, 128}].z;
    ASSERT_EQ(GRAPH_TYPE_NODE, type);

    g.derivateNode(ptr, angle::rad(0), angle::rad(0), (float)30.0, (float)1.0, 128, 128);

    type = (*g.getFramePtr())[{128, 128}].z;
    ASSERT_EQ(GRAPH_TYPE_NODE, type);

    g.acceptDerivedNodes();
    type = (*g.getFramePtr())[{128, 128}].z;
    ASSERT_EQ(GRAPH_TYPE_NODE, type);
}
