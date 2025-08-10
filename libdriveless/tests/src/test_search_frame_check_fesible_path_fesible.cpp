#include <gtest/gtest.h>
#include "../../include/search_frame.h"
#include "test_utils.h"
#include <cmath>

// FEASIBLE
// ---------------------------------------------------------------

TEST(TestSearchFrameCheckFeasiblePath, CheckPath_NoObstacles_NoPreProcess_CPU)
{
    SearchFrame f1(100, 100, {5, 5}, {15, 15});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 0.0f);
    f1.copyFrom(ptr);

    std::vector<Waypoint> path;
    for (int i = (PATH_FEASIBLE_CPU_THRESHOLD - 1); i >= 0; i--)
        path.push_back(Waypoint(50, i, angle::rad(0)));

    bool res = f1.checkFeasiblePath(path, 10, 10, true);

    ASSERT_TRUE(res);

    for (auto p : path)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCheckFeasiblePath, CheckPath_NoObstacles_PreProcessNoVectorized_CPU)
{
    SearchFrame f1(100, 100, {5, 5}, {15, 15});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 0.0f);
    f1.copyFrom(ptr);

    std::vector<Waypoint> path;
    for (int i = (PATH_FEASIBLE_CPU_THRESHOLD - 1); i >= 0; i--)
        path.push_back(Waypoint(50, i, angle::rad(0)));

    f1.processSafeDistanceZone({10, 10}, false);

    bool res = f1.checkFeasiblePath(path, 10, 10, true);

    ASSERT_TRUE(res);
    for (auto p : path)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCheckFeasiblePath, CheckPath_NoObstacles_PreProcessWithVectorized_GPU)
{
    SearchFrame f1(100, 100, {5, 5}, {15, 15});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 0.0f);
    f1.copyFrom(ptr);

    std::vector<Waypoint> path;
    for (int i = (PATH_FEASIBLE_CPU_THRESHOLD - 1); i >= 0; i--)
        path.push_back(Waypoint(50, i, angle::rad(0)));

    f1.processSafeDistanceZone({10, 10}, true);

    bool res = f1.checkFeasiblePath(path, 10, 10, true);

    ASSERT_TRUE(res);
    for (auto p : path)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCheckFeasiblePath, CheckPath_NoObstacles_NoPreProcess_GPU)
{
    SearchFrame f1(100, 100, {5, 5}, {15, 15});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 0.0f);
    f1.copyFrom(ptr);

    std::vector<Waypoint> path;
    for (int i = 2 * PATH_FEASIBLE_CPU_THRESHOLD; i >= 0; i--)
        path.push_back(Waypoint(50, i, angle::rad(0)));

    bool res = f1.checkFeasiblePath(path, 10, 10, true);

    ASSERT_TRUE(res);
    for (auto p : path)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCheckFeasiblePath, CheckPath_NoObstacles_PreProcessNoVectorized_GPU)
{
    SearchFrame f1(100, 100, {5, 5}, {15, 15});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 0.0f);
    f1.copyFrom(ptr);

    std::vector<Waypoint> path;
    for (int i = 2 * PATH_FEASIBLE_CPU_THRESHOLD; i >= 0; i--)
        path.push_back(Waypoint(50, i, angle::rad(0)));

    f1.processSafeDistanceZone({10, 10}, false);

    bool res = f1.checkFeasiblePath(path, 10, 10, true);

    ASSERT_TRUE(res);
    for (auto p : path)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCheckFeasiblePath, CheckPath_NoObstacles_PreProcessWithVectorized_CPU)
{
    SearchFrame f1(100, 100, {5, 5}, {15, 15});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);
    f1.copyFrom(ptr);

    std::vector<Waypoint> path;
    for (int i = 2 * PATH_FEASIBLE_CPU_THRESHOLD; i >= 0; i--)
        path.push_back(Waypoint(50, i, angle::rad(0)));

    f1.processSafeDistanceZone({10, 10}, true);

    bool res = f1.checkFeasiblePath(path, 10, 10, true);

    ASSERT_TRUE(res);
    for (auto p : path)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }
}
