#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include "test_utils.h"
#include "../../include/cuda_grid.h"

#define PHYS_SIZE 34.641016151377535

#define SIZE 100

extern bool check_ptr_values_int2 (int2 *ptr, int size);
extern bool check_ptr_values_float2 (float2 *ptr, int size);
extern bool check_ptr_values_double2 (double2 *ptr, int size);
extern bool check_ptr_values_int3 (int3 *ptr, int size);
extern bool check_ptr_values_float3 (float3 *ptr, int size);
extern bool check_ptr_values_double3 (double3 *ptr, int size);
extern bool check_ptr_values_int4 (int4 *ptr, int size);
extern bool check_ptr_values_float4 (float4 *ptr, int size);
extern bool check_ptr_values_double4 (double4 *ptr, int size);

TEST(TestCudaGrid, TestCudaGrid_2D)
{
    CudaGrid<int2> g_i(SIZE, SIZE);
    CudaGrid<float2> g_f(SIZE, SIZE);
    CudaGrid<double2> g_d(SIZE, SIZE);

    int CHANNELS = 2;
    float *p = new float[SIZE * SIZE * CHANNELS];

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
        {
            p[CHANNELS * (j * SIZE + i) + 0] = 1.0;
            p[CHANNELS * (j * SIZE + i) + 1] = 2.0;
        }

    g_i.copyFrom(p);
    g_f.copyFrom(p);
    g_d.copyFrom(p);

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
        {
            if (g_i[{j, i}].x != 1 || g_i[{j, i}].y != 2)
                FAIL();
            if (g_f[{j, i}].x != 1.0 || g_f[{j, i}].y != 2.0)
                FAIL();
            if (g_d[{j, i}].x != 1.0 || g_d[{j, i}].y != 2.0)
                FAIL();
        }

    int2* cudaPtr_i = g_i.getCudaPtr();

    ASSERT_TRUE(check_ptr_values_int2(g_i.getCudaPtr(), SIZE));
    ASSERT_TRUE(check_ptr_values_float2(g_f.getCudaPtr(), SIZE));
    ASSERT_TRUE(check_ptr_values_double2(g_d.getCudaPtr(), SIZE));

    g_i.clear();
    g_f.clear();
    g_d.clear();

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
        {
            if (g_i[{j, i}].x != 0 || g_i[{j, i}].y != 0)
                FAIL();
            if (g_f[{j, i}].x != 0.0 || g_f[{j, i}].y != 0.0)
                FAIL();
            if (g_d[{j, i}].x != 0.0 || g_d[{j, i}].y != 0.0)
                FAIL();
        }
}

TEST(TestCudaGrid, TestCudaGrid_3D)
{
    CudaGrid<int3> g_i(SIZE, SIZE);
    CudaGrid<float3> g_f(SIZE, SIZE);
    CudaGrid<double3> g_d(SIZE, SIZE);

    int CHANNELS = 3;
    float *p = new float[SIZE * SIZE * CHANNELS];

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
        {
            p[CHANNELS * (j * SIZE + i) + 0] = 1.0;
            p[CHANNELS * (j * SIZE + i) + 1] = 2.0;
            p[CHANNELS * (j * SIZE + i) + 2] = 3.0;
        }

    g_i.copyFrom(p);
    g_f.copyFrom(p);
    g_d.copyFrom(p);

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
        {
            if (g_i[{j, i}].x != 1 || g_i[{j, i}].y != 2 || g_i[{j, i}].z != 3)
                FAIL();
            if (g_f[{j, i}].x != 1.0 || g_f[{j, i}].y != 2.0 || g_f[{j, i}].z != 3.0)
                FAIL();
            if (g_d[{j, i}].x != 1.0 || g_d[{j, i}].y != 2.0 || g_d[{j, i}].z != 3.0)
                FAIL();
        }

    ASSERT_TRUE(check_ptr_values_int3(g_i.getCudaPtr(), SIZE));
    ASSERT_TRUE(check_ptr_values_float3(g_f.getCudaPtr(), SIZE));
    ASSERT_TRUE(check_ptr_values_double3(g_d.getCudaPtr(), SIZE));

    g_i.clear();
    g_f.clear();
    g_d.clear();

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
        {
            if (g_i[{j, i}].x != 0 || g_i[{j, i}].y != 0 || g_i[{j, i}].z != 0)
                FAIL();
            if (g_f[{j, i}].x != 0.0 || g_f[{j, i}].y != 0.0 || g_f[{j, i}].z != 0.0)
                FAIL();
            if (g_d[{j, i}].x != 0.0 || g_d[{j, i}].y != 0.0 || g_d[{j, i}].z != 0.0)
                FAIL();
        }
}

TEST(TestCudaGrid, TestCudaGrid_4D)
{
    CudaGrid<int4> g_i(SIZE, SIZE);
    CudaGrid<float4> g_f(SIZE, SIZE);
    CudaGrid<double4> g_d(SIZE, SIZE);

    int CHANNELS = 4;
    float *p = new float[SIZE * SIZE * CHANNELS];

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
        {
            p[CHANNELS * (j * SIZE + i) + 0] = 1.0;
            p[CHANNELS * (j * SIZE + i) + 1] = 2.0;
            p[CHANNELS * (j * SIZE + i) + 2] = 3.0;
            p[CHANNELS * (j * SIZE + i) + 3] = 4.0;
        }

    g_i.copyFrom(p);
    g_f.copyFrom(p);
    g_d.copyFrom(p);

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
        {
            if (g_i[{j, i}].x != 1 || g_i[{j, i}].y != 2 || g_i[{j, i}].z != 3 || g_i[{j, i}].w != 4)
                FAIL();
            if (g_f[{j, i}].x != 1.0 || g_f[{j, i}].y != 2.0 || g_f[{j, i}].z != 3.0 || g_i[{j, i}].w != 4.0)
                FAIL();
            if (g_d[{j, i}].x != 1.0 || g_d[{j, i}].y != 2.0 || g_d[{j, i}].z != 3.0 || g_i[{j, i}].w != 4.0)
                FAIL();
        }

    ASSERT_TRUE(check_ptr_values_int4(g_i.getCudaPtr(), SIZE));
    ASSERT_TRUE(check_ptr_values_float4(g_f.getCudaPtr(), SIZE));
    ASSERT_TRUE(check_ptr_values_double4(g_d.getCudaPtr(), SIZE));

    g_i.clear();
    g_f.clear();
    g_d.clear();

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
        {
            if (g_i[{j, i}].x != 0 || g_i[{j, i}].y != 0 || g_i[{j, i}].z != 0 || g_i[{j, i}].w != 0)
                FAIL();
            if (g_f[{j, i}].x != 0.0 || g_f[{j, i}].y != 0.0 || g_f[{j, i}].z != 0.0 || g_i[{j, i}].w != 0.0)
                FAIL();
            if (g_d[{j, i}].x != 0.0 || g_d[{j, i}].y != 0.0 || g_d[{j, i}].z != 0.0 || g_i[{j, i}].w != 0.0)
                FAIL();
        }
}