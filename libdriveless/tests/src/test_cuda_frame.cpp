#include <gtest/gtest.h>
#include "../../include/cuda_frame.h"
#include "test_utils.h"
#include <cmath>

TEST(TestCudaFrame, TestSetGet)
{
    CudaFrame<double3> f1(1000, 1001);
    ASSERT_EQ(1000, f1.width());
    ASSERT_EQ(1001, f1.height());
}

TEST(TestCudaFrame, TestSetAndClear)
{
    CudaFrame<double3> f1(1000, 1001);
    f1.clear();
    
    f1[{10,10}].x = 20;
    double p = f1[{10, 10}].x;
    ASSERT_FLOAT_EQ(20, p);

    f1.clear();

    p = f1[{10, 10}].x;
    ASSERT_FLOAT_EQ(0, p);
}

TEST(TestCudaFrame, TestCopyFrom)
{
    long size = 4 * 1000 * 1000;
    float *ptr = new float[size];
    for (int i = 0; i < size; i++) {
        ptr[i] = 37.7;
    }

    CudaFrame<float3> f1(1000, 1000);
    f1.copyFrom(ptr);

    for(int i = 0; i < 1000; i++)
        for(int j = 0; j < 1000; j++) {
            float x = f1[{i,j}].x;
            float y = f1[{i,j}].y;
            float z = f1[{i,j}].z;
            ASSERT_DEQ(37.7, x);
            ASSERT_DEQ(37.7, y);
            ASSERT_DEQ(37.7, z);
        }

    CudaFrame<float4> f2(1000, 1000);
    f2.copyFrom(ptr);

    for(int i = 0; i < 1000; i++)
        for(int j = 0; j < 1000; j++) {
            float x = f2[{i,j}].x;
            float y = f2[{i,j}].y;
            float z = f2[{i,j}].z;
            float w = f2[{i,j}].w;
            ASSERT_DEQ(37.7, x);
            ASSERT_DEQ(37.7, y);
            ASSERT_DEQ(37.7, z);
            ASSERT_DEQ(37.7, w);
        }

    CudaFrame<double4> f3(1000, 1000);
    f3.copyFrom(ptr);

    for(int i = 0; i < 1000; i++)
        for(int j = 0; j < 1000; j++) {
            float x = f3[{i,j}].x;
            float y = f3[{i,j}].y;
            float z = f3[{i,j}].z;
            float w = f3[{i,j}].w;
            ASSERT_DEQ(37.7, x);
            ASSERT_DEQ(37.7, y);
            ASSERT_DEQ(37.7, z);
            ASSERT_DEQ(37.7, w);
        }        

    CudaFrame<int4> f4(1000, 1000);
    f4.copyFrom(ptr);

    for(int i = 0; i < 1000; i++)
        for(int j = 0; j < 1000; j++) {
            int x = f4[{i,j}].x;
            int y = f4[{i,j}].y;
            int z = f4[{i,j}].z;
            int w = f4[{i,j}].w;
            if (x != 37 || y != 37 || z != 37 || w != 37)
                FAIL();
        }             

    CudaFrame<int2> f5(1000, 1000);
    f5.copyFrom(ptr);

    for(int i = 0; i < 1000; i++)
        for(int j = 0; j < 1000; j++) {
            int x = f4[{i,j}].x;
            int y = f4[{i,j}].y;
            if (x != 37 || y != 37)
                FAIL();
        }  
    
    CudaFrame<double2> f6(1000, 1000);
    f6.copyFrom(ptr);

    for(int i = 0; i < 1000; i++)
        for(int j = 0; j < 1000; j++) {
            float x = f3[{i,j}].x;
            float y = f3[{i,j}].y;
            ASSERT_DEQ(37.7, x);
            ASSERT_DEQ(37.7, y);
        }
}


TEST(TestCudaFrame, TestGetPointer)
{
    CudaFrame<int3> frame(1000, 1000);
    int3 *ptr = frame.getCudaPtr();

    ptr[0].x = 100;
    ptr[0].y = 101;
    ptr[0].z = 102;

    int x = frame[{0, 0}].x;
    int y = frame[{0, 0}].y;
    int z = frame[{0, 0}].z;

    ASSERT_EQ(x, 100);
    ASSERT_EQ(y, 101);
    ASSERT_EQ(z, 102);
}