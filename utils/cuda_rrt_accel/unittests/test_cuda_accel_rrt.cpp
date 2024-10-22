#include "../include/cuda_graph.h"
#include <gtest/gtest.h>

TEST(CudaRRTAccel, TestCreatingAddPointCount)
{
    CudaGraph g(1000, 1000);
    g.add_point(100, 100, 0, 0, 1000.12);
    g.add_point(100, 101, 0, 0, 1000.12);

    ASSERT_FALSE(g.checkInGraph(900, 900));
    ASSERT_TRUE(g.checkInGraph(100, 100));

    ASSERT_EQ(g.count(), 2);
}

TEST(CudaRRTAccel, TestFindBestNeighbor)
{
    CudaGraph g(1000, 1000);
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

    CudaGraph g(1000, 1000);
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