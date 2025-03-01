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
#include "../include/physical_parameters.h"
#include <vector>

int convert_to_point(int x, int z)
{
    return 256 * z + x;
}

int convert_to_key(int x, int z)
{
    return 1000 * x + z;
}

TEST(RRTGraph, TestCreateDelete)
{
    CudaGraph g(100, 100, 0, 0, 0, 0, 0, 0, 10, 10, 40, 3);
}

TEST(RRTGraph, TestBasicFeatures)
{
    CudaGraph g(100, 100, 0, 0, 0, 0, 0, 0, 10, 10, 40, 3);

    ASSERT_EQ(0, g.count());

    g.add(50, 50, 0.12, -1, -1, 0);
    g.add(50, 45, 0.12, 50, 50, 10);
    g.add(55, 40, 0.12, 50, 45, 10);
    g.add(20, 35, 0.12, 50, 45, 10);
    g.add(20, 15, 0.12, 20, 35, 10);
    g.add(20, 5, 0.12, 20, 15, 10);
    g.add(20, 0, 0.12, 20, 5, 10);

    ASSERT_EQ(7, g.count());

    g.remove(20, 0);
    ASSERT_EQ(6, g.count());

    // remove non existant should ignore
    g.remove(0, 0);
    ASSERT_EQ(6, g.count());

    // remove of out-of-bound node should ignore
    g.remove(999, -999);
    ASSERT_EQ(6, g.count());

    // does not exist
    double3 p = g.getParent(20, 0);
    ASSERT_EQ(-1, static_cast<int>(round(p.x)));
    ASSERT_EQ(-1, static_cast<int>(round(p.y)));

    // out-of-bound
    p = g.getParent(2000, -1000);
    ASSERT_EQ(-1, static_cast<int>(round(p.x)));
    ASSERT_EQ(-1, static_cast<int>(round(p.y)));

    // exists
    p = g.getParent(20, 5);
    ASSERT_EQ(20, static_cast<int>(round(p.x)));
    ASSERT_EQ(15, static_cast<int>(round(p.y)));

    // setParent of non existant node should ignore
    g.setParent(99, 99, 10, 10);
    p = g.getParent(99, 99);
    ASSERT_EQ(-1, static_cast<int>(round(p.x)));
    ASSERT_EQ(-1, static_cast<int>(round(p.y)));

    // setParent of non existant out-of-bound node should ignore
    g.setParent(999, 999, 10, 10);
    p = g.getParent(999, 999);
    ASSERT_EQ(-1, static_cast<int>(round(p.x)));
    ASSERT_EQ(-1, static_cast<int>(round(p.y)));

    // exists
    ASSERT_FLOAT_EQ(g.getCost(55, 40), 10.0);

    // doesnt exist
    ASSERT_FLOAT_EQ(g.getCost(99, 99), -1);

    // out-of-bound
    ASSERT_FLOAT_EQ(g.getCost(999, 999), -1);

    g.setCost(55, 40, 20.0);
    ASSERT_FLOAT_EQ(g.getCost(55, 40), 20.0);

    // doesnt exist
    g.setCost(99, 99, 100.0);
    ASSERT_FLOAT_EQ(g.getCost(99, 99), -1);

    // out-of-bound
    g.setCost(999, 999, 100.0);
    ASSERT_FLOAT_EQ(g.getCost(999, 999), -1);
}

TEST(RRTGraph, TestList)
{
    CudaGraph g(100, 100, 0, 0, 0, 0, 0, 0, 10, 10, 40, 3);

    ASSERT_EQ(0, g.count());
    std::unordered_map<int, double> map;

    g.add(50, 50, 0.12, -1, -1, 0);
    map[convert_to_key(50, 50)] = 0;

    g.add(50, 45, 0.12, 50, 50, 10);
    map[convert_to_key(50, 45)] = 10;

    g.add(55, 40, 0.12, 50, 45, 20);
    map[convert_to_key(55, 40)] = 20;

    g.add(20, 35, 0.12, 50, 45, 15);
    map[convert_to_key(20, 35)] = 15;

    g.add(20, 15, 0.12, 20, 35, 11);
    map[convert_to_key(20, 15)] = 11;

    g.add(20, 5, 0.12, 20, 15, 14);
    map[convert_to_key(20, 5)] = 14;

    g.add(20, 0, 0.12, 20, 5, 99);
    map[convert_to_key(20, 0)] = 99;

    int count = g.count();

    ASSERT_EQ(7, count);
    std::vector<int2> res = g.list();

    for (int i = 0; i < count; i++)
    {
        int x = static_cast<int>(round(res[i].x));
        int z = static_cast<int>(round(res[i].y));
        int key = convert_to_key(x, z);
        ASSERT_TRUE(map.find(key) != map.end());
        double heading = res[2];
        ASSERT_FLOAT_EQ(heading, 0.12);
        int px = static_cast<int>(round(res[3]));
        int pz = static_cast<int>(round(res[4]));
        double3 parent = g.getParent(x, z);
        ASSERT_EQ(px, static_cast<int>(parent.x));
        ASSERT_EQ(pz, static_cast<int>(parent.y));
        double cost = res[5];
        ASSERT_FLOAT_EQ(map[key], cost);
    }
}

TEST(RRTGraph, TestDrawPath)
{
    int c = 256 * 256 * 3;
    float *frame = new float[c];
    for (int i = 0; i < c; i++)
        frame[i] == 3;

    float rw = 256 / PHYS_SIZE;
    float rh = 256 / PHYS_SIZE;

    CudaGraph g(256, 256, PHYS_SIZE, PHYS_SIZE, 0, 0, 0, 0, rw, rh, 40, 3);

    double3 start, end;
    start.x = 128.0;
    start.y = 128.0;
    start.z = 0.0;

    end.x = 0.0;
    end.y = 0.0;
    end.z = 0.0;

    g.add(128, 128, 0.0, -1, -1, 0);

    CudaFrame f(frame, 256, 256, PHYS_SIZE, PHYS_SIZE, 0, 0, 0, 0);
    g.drawKinematicPath(f.getFramePtr(), start, end);
    TestFrame::dump_cuda_frame_to_file(&f, "dump_gpu.png");
}

TEST(RRTGraph, TestDrawPathCPU)
{
    int c = 256 * 256 * 3;
    float *frame = new float[c];
    for (int i = 0; i < c; i++)
        frame[i] == 3;

    float rw = 256 / PHYS_SIZE;
    float rh = 256 / PHYS_SIZE;

    CudaGraph g(256, 256, PHYS_SIZE, PHYS_SIZE, 0, 0, 0, 0, rw, rh, 40, 3);

    double3 start, end;
    start.x = 128.0;
    start.y = 128.0;
    start.z = 0.0;

    end.x = 0.0;
    end.y = 0.0;
    end.z = 0.0;

    g.add(128, 128, 0.0, -1, -1, 0);

    CudaFrame f(frame, 256, 256, PHYS_SIZE, PHYS_SIZE, 0, 0, 0, 0);

    std::vector<double3> list = CurveGenerator::buildCurveWaypoints(start, rw, rh, 3, 40, start, end, 1);
    float3 *ptr = f.getFramePtr();

    for (double3 p : list)
    {
        int pos = p.y * 256 + p.x;
        if (pos >= 256 * 256 || pos < 0)
            continue;
        ptr[pos].x = 255;
        ptr[pos].y = 255;
        ptr[pos].z = 255;
    }

    TestFrame::dump_cuda_frame_to_file(&f, "dump_cpu.png");
}
