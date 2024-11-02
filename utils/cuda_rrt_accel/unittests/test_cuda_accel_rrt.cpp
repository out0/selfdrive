#include "../include/cuda_graph.h"
#include <gtest/gtest.h>

TEST(CudaRRTAccel, TestCreatingAddPointCount)
{
    CudaGraph g(1000, 1000, 0, 0, 0, 0, 0, 0);
    g.add_point(100, 100, 0, 0, 1000.12);
    g.add_point(100, 101, 0, 0, 1000.12);

    ASSERT_FALSE(g.checkInGraph(900, 900));
    ASSERT_TRUE(g.checkInGraph(100, 100));

    ASSERT_EQ(g.count(), 2);
}

TEST(CudaRRTAccel, TestFindBestNeighbor)
{
    CudaGraph g(1000, 1000, 0, 0, 0, 0, 0, 0);
    g.add_point(100, 100, -1, -1, 0);
    g.add_point(100, 70, -1, -1, 0);
    g.add_point(130, 50, -1, -1, 0);

    int *res = g.find_best_neighbor(110, 100, 1.0);

    ASSERT_EQ(res[0], 0);
    ASSERT_EQ(res[1], 0);
    ASSERT_EQ(res[2], 0);

    res = g.find_best_neighbor(110, 100, 10.0);
    ASSERT_EQ(res[0], 100);
    ASSERT_EQ(res[1], 100);
    ASSERT_EQ(res[2], 1);
    delete[] res;

    res = g.find_best_neighbor(120, 80, 1000.0);
    ASSERT_EQ(res[0], 100);
    ASSERT_EQ(res[1], 70);
    ASSERT_EQ(res[2], 1);
    delete[] res;
}

TEST(CudaRRTAccel, TestGetParent)
{

    CudaGraph g(1000, 1000, 0, 0, 0, 0, 0, 0);
    g.add_point(100, 100, -1, -1, 0);
    g.add_point(100, 70, 100, 100, 0);
    g.add_point(130, 50, 100, 70, 0);

    int *res = g.getParent(0, 0);
    ASSERT_EQ(res[0], 0);
    ASSERT_EQ(res[1], 0);
    ASSERT_EQ(res[2], 0);

    res = g.getParent(100, 100);
    ASSERT_EQ(res[0], -1);
    ASSERT_EQ(res[1], -1);
    ASSERT_EQ(res[2], 1);
    delete[] res;

    res = g.getParent(100, 70);
    ASSERT_EQ(res[0], 100);
    ASSERT_EQ(res[1], 100);
    ASSERT_EQ(res[2], 1);
    delete[] res;

    res = g.getParent(130, 50);
    ASSERT_EQ(res[0], 100);
    ASSERT_EQ(res[1], 70);
    ASSERT_EQ(res[2], 1);
    delete[] res;
}

TEST(CudaRRTAccel, TestListNodes)
{
    CudaGraph g(1000, 1000, 0, 0, 0, 0, 0, 0);
    g.add_point(100, 100, -1, -1, 0);
    g.add_point(100, 70, 100, 100, 10);
    g.add_point(130, 50, 100, 70, 20);
    g.add_point(140, 30, 130, 50, 30);
    g.add_point(160, 10, 100, 70, 40);
    g.add_point(190, 0, 100, 70, 50);
    g.add_point(210, 0, 100, 70, 50);

    int count = g.count();

    ASSERT_EQ(7, count);

    float *nodes = new float[5 * count];
    g.listNodes(nodes, count);

    for (int i = 0; i < count; i++)
    {
        printf("(%d, %d):   parent (%d, %d),  total cost: %f\n",
         static_cast<int>(round(nodes[5*i])),
         static_cast<int>(round(nodes[5*i+1])),
         static_cast<int>(round(nodes[5*i+2])),
         static_cast<int>(round(nodes[5*i+3])),
         nodes[5*i+4]);
    }

    delete[] nodes;
}