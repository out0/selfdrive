#include "../src/cuda_graph.h"
#include "../src/cuda_frame.h"
#include "../src/kinematic_model.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "test_frame.h"
#include <thread>
#include <chrono>
#include <unordered_map>
#include "test_frame.h"

TEST(RRTGraph, TestNearestNeighboor)
{
    CudaGraph g(100, 100, 0, 0, 0, 0, 0, 0, 10, 10, 40, 3, 1);

    ASSERT_EQ(0, g.count());

    g.add(50, 50, 0.12, -1, -1, 0);
    g.add(50, 45, 0.12, 50, 50, 10);
    g.add(55, 40, 0.12, 50, 45, 20);
    g.add(20, 35, 0.12, 50, 45, 15);
    g.add(20, 15, 0.12, 20, 35, 11);
    g.add(20, 5, 0.12, 20, 15, 14);
    g.add(20, 0, 0.12, 20, 5, 99);

    int2 res = g.find_nearest_neighbor(50, 50);
    ASSERT_EQ(res.x, 50);
    ASSERT_EQ(res.y, 50);

    res = g.find_nearest_neighbor(50, 44);
    ASSERT_EQ(res.x, 50);
    ASSERT_EQ(res.y, 45);

    res = g.find_nearest_neighbor(21, 12);
    ASSERT_EQ(res.x, 20);
    ASSERT_EQ(res.y, 15);

    res = g.find_nearest_neighbor(21, -12);
    ASSERT_EQ(res.x, 20);
    ASSERT_EQ(res.y, 0);
}

TEST(RRTGraph, TestNearestFeasibleNeighboor_NoObstacles)
{
    TestFrame tstFrame;
    CudaGraph *g = tstFrame.getGraph();
    CudaFrame *og = tstFrame.getCudaGrame();

    g->add(128, 128, 0.0, -1, -1, 0);
    g->add(128, 108, 0.0, 128, 128, 0);
    g->add(128, 88, 0.0, 128, 108, 0);
    g->add(128, 48, 0.0, 128, 88, 0);
    g->add(150, 0, 0.0, 128, 48, 0);
    
    int2 res = g->find_nearest_feasible_neighbor(og->getFramePtr(), 50, 50);
    ASSERT_EQ(res.x, 128);
    ASSERT_EQ(res.y, 88);

    res = g->find_nearest_feasible_neighbor(og->getFramePtr(), 150, 20);
    ASSERT_EQ(res.x, 128);
    ASSERT_EQ(res.y, 48);

    res = g->find_nearest_feasible_neighbor(og->getFramePtr(), 100, 70);
    ASSERT_EQ(res.x, 128);
    ASSERT_EQ(res.y, 88);

    res = g->find_nearest_feasible_neighbor(og->getFramePtr(), 140, -20);
    ASSERT_EQ(res.x, 150);
    ASSERT_EQ(res.y, 0);

    // tstFrame.drawGraph();
    // tstFrame.toFile();
}

TEST(RRTGraph, TestNearestFeasibleNeighboor_Obstacle)
{
    TestFrame tstFrame;
    CudaGraph *g = tstFrame.getGraph();
    CudaFrame *og = tstFrame.getCudaGrame();

    g->add(128.0, 128.0, 0.0, -1, -1, 0);
    g->add(128.0, 108.0, 0.0, 128.0, 128.0, 0);
    g->add(128.0, 88.0, 0.0, 128.0, 108.0, 0);
    g->add(128.0, 48.0, 0.0, 128.0, 88.0, 0);
    g->add(150.0, 0.0, 0.0, 128.0, 48.0, 0);

    tstFrame.addArea(130, 90, 140, 70, 28);
    // tstFrame.drawGraph();
    // tstFrame.toFile("dump_obstacle.png");

    int2 res = g->find_nearest_feasible_neighbor(og->getFramePtr(), 50, 50);
    ASSERT_EQ(res.x, 128);
    ASSERT_EQ(res.y, 128);

    res = g->find_nearest_feasible_neighbor(og->getFramePtr(), 150, 20);
    ASSERT_EQ(res.x, 128);
    ASSERT_EQ(res.y, 48);

    res = g->find_nearest_feasible_neighbor(og->getFramePtr(), 100, 70);
    ASSERT_EQ(res.x, -1);
    ASSERT_EQ(res.y, -1);

    res = g->find_nearest_feasible_neighbor(og->getFramePtr(), 140, -20);
    ASSERT_EQ(res.x, 150);
    ASSERT_EQ(res.y, 0);

    // g->add(50, 50, -20, 128, 128, 200);
    // g->add(150, 20, -20, 128, 48, 200);


    // tstFrame.drawGraph();
    // tstFrame.toFile("dump_obstacle.png");

}
