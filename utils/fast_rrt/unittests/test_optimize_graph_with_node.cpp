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



TEST(RRTGraph, TestOptimize_NoObstacle)
{
    TestFrame tstFrame;
    CudaGraph *g = tstFrame.getGraph();
    CudaFrame *og = tstFrame.getCudaGrame();

    g->add(128, 230, 0.0, -1, -1, 0);



    g->add(128, 108, 0.0, 128, 230, 0);
    g->add(128, 88, 0.0, 128, 108, 0);
    g->add(128, 48, 0.0, 128, 88, 0);
    g->add(150, 0, 0.0, 128, 48, 0);
    
    tstFrame.drawGraph();
    tstFrame.toFile("graph_optim.png");
}
