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

       //tstFrame.addArea(130, 90, 140, 70, 28);

    g->add(128, 230, 0.0, -1, -1, 0);

    int2 n1 = g->deriveNode(og->getFramePtr(), 128, 230, -30, 100);
    //ASSERT_NE(-1, n1.x);

    int2 n2 = g->deriveNode(og->getFramePtr(), n1.x, n1.y, 30, 100);
    //ASSERT_NE(-1, n2.x);
    //ASSERT_TRUE(g->connectToGraph(og->getFramePtr(), 128, 230, 100, 200));

    int2 n3 = g->deriveNode(og->getFramePtr(), n2.x, n2.y, 35, 50);
    //ASSERT_NE(-1, n3.x);
    
    int2 n4a = g->deriveNode(og->getFramePtr(), n3.x, n3.y, 35, 50);
    //ASSERT_NE(-1, n4a.x);
       
    int2 n4b = g->deriveNode(og->getFramePtr(), n3.x, n3.y, -15, 50);
    //ASSERT_NE(-1, n4b.x);
    
    // 128 não aceita. Pq??
    g->connectToGraph(og->getFramePtr(), n4b.x, n4b.y, 118, 0);

    // g->add(128, 108, 0.0, 128, 230, 0);
    // g->add(128, 88, 0.0, 128, 108, 0);
    // g->add(128, 48, 0.0, 128, 88, 0);
    // g->add(150, 0, 0.0, 128, 48, 0);
    
    tstFrame.addArea(130, 90, 140, 70, 28);

    tstFrame.drawGraph();
    tstFrame.toFile("graph_optim.png");
}
