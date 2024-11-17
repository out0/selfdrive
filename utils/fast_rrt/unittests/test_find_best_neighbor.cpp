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
#include <math.h>

float compute_distances(int x1, int z1, int x2, int z2)
{
    int dx = x2 - x1;
    int dz = z2 - z1;
    return sqrt(dx * dx + dz * dz);
}

TEST(RRTGraph, TestBestFeasibleNeighboor_NoObstacles_NoCost)
{
    TestFrame tstFrame;
    CudaGraph *g = tstFrame.getGraph();
    CudaFrame *og = tstFrame.getCudaGrame();

    g->add(128, 128, 0.0, -1, -1, 0);
    g->add(128, 108, 0.0, 128, 128, 0);
    g->add(128, 88, 0.0, 128, 108, 0);
    g->add(128, 48, 0.0, 128, 88, 0);
    g->add(150, 0, 0.0, 128, 48, 0);

    int2 res = g->find_best_feasible_neighbor(og->getFramePtr(), 50, 50, 300);
    ASSERT_EQ(res.x, 128);
    ASSERT_EQ(res.y, 88);

    res = g->find_best_feasible_neighbor(og->getFramePtr(), 150, 20, 300);
    ASSERT_EQ(res.x, 128);
    ASSERT_EQ(res.y, 48);

    res = g->find_best_feasible_neighbor(og->getFramePtr(), 100, 70, 300);
    ASSERT_EQ(res.x, 128);
    ASSERT_EQ(res.y, 88);

    res = g->find_best_feasible_neighbor(og->getFramePtr(), 140, -20, 300);
    ASSERT_EQ(res.x, 150);
    ASSERT_EQ(res.y, 0);

    // tstFrame.drawGraph();
    // tstFrame.toFile();
}

TEST(RRTGraph, TestBestFeasibleNeighboor)
{
    TestFrame tstFrame;
    CudaGraph *g = tstFrame.getGraph();
    CudaFrame *og = tstFrame.getCudaGrame();

    g->add(128.0, 128.0, 0.0, -1, -1, 0);
    g->add(128.0, 108.0, 0.0, 128.0, 128.0, 0);
    g->add(128.0, 88.0, 0.0, 128.0, 108.0, 0);
    g->add(128.0, 48.0, 0.0, 128.0, 88.0, 0);
    g->add(128.0, 44.0, 0.0, 128.0, 48.0, 100);    
    g->add(150.0, 0.0, 0.0, 128.0, 48.0, 0);

    tstFrame.addArea(130, 90, 140, 70, 28);
    // tstFrame.drawGraph();
    // tstFrame.toFile("dump_obstacle.png");

    int2 res = g->find_best_feasible_neighbor(og->getFramePtr(), 50, 50, 300);
    ASSERT_EQ(res.x, 128);
    ASSERT_EQ(res.y, 128);

    res = g->find_best_feasible_neighbor(og->getFramePtr(), 150, 20, 300);
    ASSERT_EQ(res.x, 128);
    ASSERT_EQ(res.y, 48);  // it choses 128,48 because 128,44 has +100 of cost

    res = g->find_best_feasible_neighbor(og->getFramePtr(), 100, 70, 300);
    ASSERT_EQ(res.x, -1);
    ASSERT_EQ(res.y, -1);

    res = g->find_best_feasible_neighbor(og->getFramePtr(), 140, -20, 300);
    ASSERT_EQ(res.x, 150);
    ASSERT_EQ(res.y, 0);
}

// TEST(RRTGraph, TestBestFeasibleNeighboor_Obstacle)
// {
//     TestFrame tstFrame;
//     CudaGraph *g = tstFrame.getGraph();
//     CudaFrame *og = tstFrame.getCudaGrame();

//     g->add(128.0, 128.0, 0.0, -1, -1, 0);
//     g->add(128.0, 108.0, 0.0, 128.0, 128.0, 0);
//     g->add(128.0, 88.0, 0.0, 128.0, 108.0, 0);
//     g->add(128.0, 48.0, 0.0, 128.0, 88.0, 0);
//     g->add(150.0, 0.0, 0.0, 128.0, 48.0, 0);

//     tstFrame.addArea(130, 90, 140, 70, 28);
//     // tstFrame.drawGraph();
//     // tstFrame.toFile("dump_obstacle.png");

//     int2 res = g->find_nearest_feasible_neighbor(og->getFramePtr(), 50, 50);
//     ASSERT_EQ(res.x, 128);
//     ASSERT_EQ(res.y, 128);

//     res = g->find_nearest_feasible_neighbor(og->getFramePtr(), 150, 20);
//     ASSERT_EQ(res.x, 128);
//     ASSERT_EQ(res.y, 48);

//     res = g->find_nearest_feasible_neighbor(og->getFramePtr(), 100, 70);
//     ASSERT_EQ(res.x, -1);
//     ASSERT_EQ(res.y, -1);

//     res = g->find_nearest_feasible_neighbor(og->getFramePtr(), 140, -20);
//     ASSERT_EQ(res.x, 150);
//     ASSERT_EQ(res.y, 0);

//     // g->add(50, 50, -20, 128, 128, 200);
//     // g->add(150, 20, -20, 128, 48, 200);

//     // tstFrame.drawGraph();
//     // tstFrame.toFile("dump_obstacle.png");

// }
