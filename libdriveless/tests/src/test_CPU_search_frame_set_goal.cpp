#include <gtest/gtest.h>
#include "../../include/search_frame_cpu.h"
#include "test_utils_cpu.h"
#include <cmath>


TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZone_NoObstacles)
{
    SearchFrameCPU f1(100, 100, {5, 5}, {15, 15});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, false);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (!f1.isTraversable(x, z))
            {
                printf("it should be traversable at (%d, %d)\n", x, z);
                FAIL();
            }
}

void setClassValueCPU(float *ptr, int width, int x, int z, int value)
{
    int pos = 3 * (width * z + x);
    ptr[pos] = value;
}


TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZoneSingleObstacle)
{
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);

    // adding obstacle to (50, 50)
    setClassValueCPU(ptr, 100, 50, 50, 1);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, false);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (x >= 43 && x <= 57 && z >= 43 && z <= 57)
            {
                if (f1.isTraversable(x, z))
                {
                    printf("it should NOT be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
            else
            {
                if (!f1.isTraversable(x, z))
                {
                    printf("it should be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
}

TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZoneTwoPixelZLineObstacle) {
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);

    // adding obstacle to (50, 50)
    setClassValueCPU(ptr, 100, 50, 50, 1);
    setClassValueCPU(ptr, 100, 50, 51, 1);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, false);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (x >= 43 && x <= 57 && z >= 43 && z <= 58)
            {
                if (f1.isTraversable(x, z))
                {
                    printf("it should NOT be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
            else
            {
                if (!f1.isTraversable(x, z))
                {
                    printf("it should be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
}

TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZoneThreePixelZLineObstacle)
{
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);

    // adding obstacle to (50, 50)
    setClassValueCPU(ptr, 100, 50, 50, 1);
    setClassValueCPU(ptr, 100, 50, 51, 1);
    setClassValueCPU(ptr, 100, 50, 52, 1);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, false);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (x >= 43 && x <= 57 && z >= 43 && z <= 59)
            {
                if (f1.isTraversable(x, z))
                {
                    printf("it should NOT be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
            else
            {
                if (!f1.isTraversable(x, z))
                {
                    printf("it should be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
}

TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZoneTwoPixelXLineObstacle) {
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);

    // adding obstacle to (50, 50)
    setClassValueCPU(ptr, 100, 50, 50, 1);
    setClassValueCPU(ptr, 100, 49, 50, 1);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, false);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (x >= 42 && x <= 57 && z >= 43 && z <= 57)
            {
                if (f1.isTraversable(x, z))
                {
                    printf("it should NOT be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
            else
            {
                if (!f1.isTraversable(x, z))
                {
                    printf("it should be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
}

TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZoneThreePixelXLineObstacle)
{
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);

    // adding obstacle to (50, 50)
    setClassValueCPU(ptr, 100, 50, 50, 1);
    setClassValueCPU(ptr, 100, 49, 50, 1);
    setClassValueCPU(ptr, 100, 51, 50, 1);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, false);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (x >= 42 && x <= 58 && z >= 43 && z <= 57)
            {
                if (f1.isTraversable(x, z))
                {
                    printf("it should NOT be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
            else
            {
                if (!f1.isTraversable(x, z))
                {
                    printf("it should be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
}

TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZoneFatObstacle)
{
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);

    // adding obstacle to (43-48, 43-48)
    int obst_init = 43;
    int obst_end = 48;
    for (int z = obst_init; z <= obst_end; z++)
        for (int x = obst_init; x <= obst_end; x++)
            setClassValueCPU(ptr, 100, x, z, 1);
    f1.copyFrom(ptr);
    f1.processSafeDistanceZone({10, 10}, false);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (x >= (obst_init-7) && x <= (obst_end+7) && z >= (obst_init-7) && z <= (obst_end+7))
            {
                if (f1.isTraversable(x, z))
                {
                    printf("it should NOT be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
            else
            {
                if (!f1.isTraversable(x, z))
                {
                    printf("it should be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
}

TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZoneSingleObstacle_WithVectorizeFlag)
{
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);

    // adding obstacle to (50, 50)
    setClassValueCPU(ptr, 100, 50, 50, 1);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, true);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (x >= 43 && x <= 57 && z >= 43 && z <= 57)
            {
                if (f1.isTraversable(x, z))
                {
                    printf("it should NOT be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
            else
            {
                if (!f1.isTraversable(x, z))
                {
                    printf("it should be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
}

TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZoneTwoPixelZLineObstacle_WithVectorizeFlag) {
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);

    // adding obstacle to (50, 50)
    setClassValueCPU(ptr, 100, 50, 50, 1);
    setClassValueCPU(ptr, 100, 50, 51, 1);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, true);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (x >= 43 && x <= 57 && z >= 43 && z <= 58)
            {
                if (f1.isTraversable(x, z))
                {
                    printf("it should NOT be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
            else
            {
                if (!f1.isTraversable(x, z))
                {
                    printf("it should be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
}

TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZoneThreePixelZLineObstacle_WithVectorizeFlag)
{
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);

    // adding obstacle to (50, 50)
    setClassValueCPU(ptr, 100, 50, 50, 1);
    setClassValueCPU(ptr, 100, 50, 51, 1);
    setClassValueCPU(ptr, 100, 50, 52, 1);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, true);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (x >= 43 && x <= 57 && z >= 43 && z <= 59)
            {
                if (f1.isTraversable(x, z))
                {
                    printf("it should NOT be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
            else
            {
                if (!f1.isTraversable(x, z))
                {
                    printf("it should be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
}

TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZoneTwoPixelXLineObstacle_WithVectorizeFlag) {
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);

    // adding obstacle to (50, 50)
    setClassValueCPU(ptr, 100, 50, 50, 1);
    setClassValueCPU(ptr, 100, 49, 50, 1);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, true);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (x >= 42 && x <= 57 && z >= 43 && z <= 57)
            {
                if (f1.isTraversable(x, z))
                {
                    printf("it should NOT be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
            else
            {
                if (!f1.isTraversable(x, z))
                {
                    printf("it should be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
}

TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZoneThreePixelXLineObstacle_WithVectorizeFlag)
{
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);

    // adding obstacle to (50, 50)
    setClassValueCPU(ptr, 100, 50, 50, 1);
    setClassValueCPU(ptr, 100, 49, 50, 1);
    setClassValueCPU(ptr, 100, 51, 50, 1);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, true);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (x >= 42 && x <= 58 && z >= 43 && z <= 57)
            {
                if (f1.isTraversable(x, z))
                {
                    printf("it should NOT be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
            else
            {
                if (!f1.isTraversable(x, z))
                {
                    printf("it should be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
}

TEST(TestSearchFrameCPUprocessSafeDistanceZone, TestprocessSafeDistanceZoneFatObstacle_WithVectorizeFlag)
{
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);

    // adding obstacle to (43-48, 43-48)
    int obst_init = 43;
    int obst_end = 48;
    for (int z = obst_init; z <= obst_end; z++)
        for (int x = obst_init; x <= obst_end; x++)
            setClassValueCPU(ptr, 100, x, z, 1);
    f1.copyFrom(ptr);
    f1.processSafeDistanceZone({10, 10}, true);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++)
            if (x >= (obst_init-7) && x <= (obst_end+7) && z >= (obst_init-7) && z <= (obst_end+7))
            {
                if (f1.isTraversable(x, z))
                {
                    printf("it should NOT be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
            else
            {
                if (!f1.isTraversable(x, z))
                {
                    printf("it should be traversable at (%d, %d)\n", x, z);
                    FAIL();
                }
            }
}

