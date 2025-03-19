#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <unordered_map>
#include "../../../cudac/include/cuda_frame.h"
#include <cmath>
#include "test_utils.h"

#define PHYS_SIZE 34.641016151377535

TEST(TestSimpleAPF, TestAPF)
{
    CudaGraph g(256, 256);
    float3 *ptr = createEmptySearchFrame(256, 256);
    angle maxSteering = angle::deg(40);
    int costs[] = {{0},
                   {1},
                   {-1},
                   {3},
                   {4},
                   {5}};
    g.setPhysicalParams(PHYS_SIZE, PHYS_SIZE, maxSteering, 5.412658773);
    g.setClassCosts(costs, 6);
    g.setSearchParams({0, 0}, {-1, -1}, {-1, -1});
    g.add(128, 230, angle::rad(0.0), -1, -1, 0);


    float3 *og = createEmptySearchFrame(256, 256);
    og[128 * 256 + 128].x = 2; // single obstacle in 128,128
    g.computeAPF(og, 5);
    
}