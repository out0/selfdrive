
#pragma once

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "../src/cuda_frame.h"
#include "../src/cuda_graph.h"
#include <unordered_set>
#include <math.h>

#define OG_REAL_WIDTH 34.641016151377535
#define OG_REAL_HEIGHT 34.641016151377535
#define MAX_STEERING_ANGLE 40

class TestFrame
{
private:
    CudaFrame *og;
    CudaGraph *graph;
    int toSetKey(int x, int z);
    void drawNode(CudaGraph *graph, float3 *imgPtr, int x, int z, double heading);
    void drawNodeDebug(CudaGraph *graph, float3 *imgPtr, int3 *imgPtrOutput,  int x, int z, double heading);
    double _rw;
    double _rh;
    double _lr;


public:
    TestFrame(int default_fill_value = 1);
    ~TestFrame();
    void toFile(const char *filename);
    CudaGraph *getGraph();
    CudaFrame *getCudaGrame();
    void addArea(int x1, int z1, int x2, int z2, int classType);
    void drawGraph();
    void drawGraphDebugTo(const char *file);

    static void dump_cuda_frame_to_file(CudaFrame *frame, const char *filename);
};

