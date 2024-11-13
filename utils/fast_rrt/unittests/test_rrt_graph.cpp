#include "../src/cuda_graph.h"
#include "../src/cuda_frame.h"
#include "../src/kinematic_model.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "test_utils.h"
#include <thread>
#include <chrono>

TEST(RRTGraph, TestCreateDelete)
{
    CudaGraph *g = new CudaGraph(100, 100, 0, 0, 0, 0, 0, 0, 10, 10, 40, 3, 1);
    delete g;
}

TEST(RRTGraph, TestBasicFeatures)
{
    CudaGraph *g = new CudaGraph(100, 100, 0, 0, 0, 0, 0, 0, 10, 10, 40, 3, 1);

    ASSERT_EQ(0, g->count());

    g->add(50, 50, -1, -1, 0);
    g->add(50, 45, 50, 50, 10);
    g->add(55, 40, 50, 45, 10);
    g->add(20, 35, 50, 45, 10);
    g->add(20, 15, 20, 35, 10);
    g->add(20, 5, 20, 15, 10);
    g->add(20, 0, 20, 5, 10);

    ASSERT_EQ(7, g->count());

    g->remove(20, 0);
    ASSERT_EQ(6, g->count());

    // remove non existant should ignore
    g->remove(0, 0);
    ASSERT_EQ(6, g->count());

    // remove of out-of-bound node should ignore
    g->remove(999, -999);
    ASSERT_EQ(6, g->count());

    // does not exist
    int2 p = g->getParent(20, 0);
    ASSERT_EQ(-1, p.x);
    ASSERT_EQ(-1, p.y);

    // out-of-bound
    p = g->getParent(2000, -1000);
    ASSERT_EQ(-1, p.x);
    ASSERT_EQ(-1, p.y);

    // exists
    p = g->getParent(20, 5);
    ASSERT_EQ(20, p.x);
    ASSERT_EQ(15, p.y);

    // setParent of non existant node should ignore
    g->setParent(99, 99, 10, 10);
    p = g->getParent(99, 99);
    ASSERT_EQ(-1, p.x);
    ASSERT_EQ(-1, p.y);

    // setParent of non existant out-of-bound node should ignore
    g->setParent(999, 999, 10, 10);
    p = g->getParent(999, 999);
    ASSERT_EQ(-1, p.x);
    ASSERT_EQ(-1, p.y);

    // exists
    ASSERT_FLOAT_EQ(g->getCost(55, 40), 10.0);

    // doesnt exist
    ASSERT_FLOAT_EQ(g->getCost(99, 99), -1);

    // out-of-bound
    ASSERT_FLOAT_EQ(g->getCost(999, 999), -1);

    g->setCost(55, 40, 20.0);
    ASSERT_FLOAT_EQ(g->getCost(55, 40), 20.0);

    // doesnt exist
    g->setCost(99, 99, 100.0);
    ASSERT_FLOAT_EQ(g->getCost(99, 99), -1);

    // out-of-bound
    g->setCost(999, 999, 100.0);
    ASSERT_FLOAT_EQ(g->getCost(999, 999), -1);

    delete g;
}

#define PHYS_SIZE 34.641016151377535

TEST(RRTGraph, TestCudaPathCPUPathSameBehaviorTOP)
{
    int c = 256 * 256 * 3;
    float *frame = new float[c];
    for (int i = 0; i < c; i++)
        frame[i] == 3;

    float rw = 256 / PHYS_SIZE;
    float rh = 256 / PHYS_SIZE;

    CudaGraph *g = new CudaGraph(256, 256, PHYS_SIZE, PHYS_SIZE, 0, 0, 0, 0, rw, rh, 40, 3, 1);

    float3 start, end;
    start.x = 128.0;
    start.y = 128.0;
    start.z = 0.0;
    end.x = 0.0;
    end.y = 0.0;
    end.z = 0.0;

    CurveGenerator gen(start, rw, rh, 3, 40);
    g->add(128, 128, -1, -1, 0);

    for (int x = 0; x < 1; x++) {
        end.x = x;
        CudaFrame *f = new CudaFrame(frame, 256, 256, PHYS_SIZE, PHYS_SIZE, 0, 0, 0, 0);
        float3 *ptrGpu = f->getFramePtr();
        g->drawKinematicPath(ptrGpu, start, end);
        Memlist<float3> *listCpu = gen.buildCurveWaypoints(start, end, 1);

        for (int i = 0; i < listCpu->size; i++) {
            float3 p = listCpu->data[i];
            int pos = p.y * 256 + p.x;
            if (ptrGpu[pos].x != 255) {
                //printf("[%d] invalid GPU pos: %d, %d\n", i, static_cast<int>(p.x), static_cast<int>(p.y));
                //FAIL();
            }
            
        }

        delete listCpu;
        delete f;
    }


    
}
TEST(RRTGraph, TestDrawPath)
{
    return;
    int c = 256 * 256 * 3;
    float *frame = new float[c];
    for (int i = 0; i < c; i++)
        frame[i] == 3;

    float rw = 256 / PHYS_SIZE;
    float rh = 256 / PHYS_SIZE;

    CudaGraph *g = new CudaGraph(256, 256, PHYS_SIZE, PHYS_SIZE, 0, 0, 0, 0, rw, rh, 40, 3, 1);

    float3 start, end;
    start.x = 128.0;
    start.y = 128.0;
    start.z = 0.0;

    end.x = 0.0;
    end.y = 0.0;
    end.z = 0.0;

    g->add(128, 128, -1, -1, 0);

    /*
    for (int x = 200; x < 256; x++) {
        end.x = x;
        CudaFrame *f = new CudaFrame(frame, 256, 256, PHYS_SIZE, PHYS_SIZE, 0, 0, 0, 0);
        g->drawKinematicPath(f->getFramePtr(), start, end);
        dump_cuda_frame_to_file(f, "dump.png");
        delete f;
        printf("(x, z) = %d, %d\n", x, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    */
    end.x = 230;
    end.y = 230;
    CudaFrame *f = new CudaFrame(frame, 256, 256, PHYS_SIZE, PHYS_SIZE, 0, 0, 0, 0);
    g->drawKinematicPath(f->getFramePtr(), start, end);
    dump_cuda_frame_to_file(f, "dump.png");
    delete f;
}

TEST(RRTGraph, TestDrawPathCPU)
{
    return;
    int c = 256 * 256 * 3;
    float *frame = new float[c];
    for (int i = 0; i < c; i++)
        frame[i] == 3;

    float rw = 256 / PHYS_SIZE;
    float rh = 256 / PHYS_SIZE;

    CudaGraph *g = new CudaGraph(256, 256, PHYS_SIZE, PHYS_SIZE, 0, 0, 0, 0, rw, rh, 40, 3, 1);

    float3 start, end;
    start.x = 128.0;
    start.y = 128.0;
    start.z = 180.0;

    end.x = 255.0;
    end.y = 255.0;
    end.z = 0.0;

    g->add(128, 128, -1, -1, 0);

    CurveGenerator gen(start, rw, rh, 3, 40);
    CudaFrame *f = new CudaFrame(frame, 256, 256, PHYS_SIZE, PHYS_SIZE, 0, 0, 0, 0);

    Memlist<float3> *list = gen.buildCurveWaypoints(start, end, 1);
    float3 *ptr = f->getFramePtr();
    for (int i = 0; i < list->size; i++)
    {
        float3 p = list->data[i];
        int pos = p.y * 256 + p.x;
        ptr[pos].x = 255;
        ptr[pos].y = 255;
        ptr[pos].z = 255;
    }

    dump_cuda_frame_to_file(f, "dump.png");
    delete f;
    delete list;
}