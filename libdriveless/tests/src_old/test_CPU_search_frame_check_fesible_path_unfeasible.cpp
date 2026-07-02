#include <gtest/gtest.h>
#include "../../include/search_frame_cpu.h"
#include "test_utils_cpu.h"
#include <cmath>

// UNFEASIBLE
// ---------------------------------------------------------------

TEST(TestSearchFrameCPUCheckFeasiblePath, CheckPath_Unfeasible_NoPreProcess_CPU)
{
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 1.0f);
    f1.copyFrom(ptr);

    std::vector<Waypoint> path;
    for (int i = (PATH_FEASIBLE_CPU_THRESHOLD - 1); i >= 0; i--)
        path.push_back(Waypoint(50, i, angle::rad(0)));

    bool res = f1.checkFeasiblePath(path, 10, 10, true);

    ASSERT_FALSE(res);

    for (auto p : path)
    {
        if (p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCPUCheckFeasiblePath, CheckPath_Unfeasible_PreProcessNoVectorized_CPU)
{
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 1.0f);
    f1.copyFrom(ptr);

    std::vector<Waypoint> path;
    for (int i = (PATH_FEASIBLE_CPU_THRESHOLD - 1); i >= 0; i--)
        path.push_back(Waypoint(50, i, angle::rad(0)));

    f1.processSafeDistanceZone({10, 10}, false);

    bool res = f1.checkFeasiblePath(path, 10, 10, true);

    ASSERT_FALSE(res);

    for (auto p : path)
    {
        if (p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCPUCheckFeasiblePath, CheckPath_Unfeasible_PreProcessWithVectorized_CPU)
{
    SearchFrameCPU f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 1.0f);
    f1.copyFrom(ptr);

    std::vector<Waypoint> path;
    for (int i = (PATH_FEASIBLE_CPU_THRESHOLD - 1); i >= 0; i--)
        path.push_back(Waypoint(50, i, angle::rad(0)));

    f1.processSafeDistanceZone({10, 10}, true);

    bool res = f1.checkFeasiblePath(path, 10, 10, true);

    ASSERT_FALSE(res);

    for (auto p : path)
    {
        if (p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCPUCheckFeasiblePath, CheckPath_Unfeasible_NoPreProcess_GPU)
{
    SearchFrameCPU f1(100, 100, {5, 5}, {15, 15});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 1.0f);
    f1.copyFrom(ptr);

    std::vector<Waypoint> path;
    for (int i = 2 * PATH_FEASIBLE_CPU_THRESHOLD; i >= 0; i--)
        path.push_back(Waypoint(50, i, angle::rad(0)));

    bool res = f1.checkFeasiblePath(path, 10, 10, true);

    ASSERT_FALSE(res);

    for (auto p : path)
    {
        if (p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCPUCheckFeasiblePath, CheckPath_Unfeasible_PreProcessNoVectorized_GPU)
{
    SearchFrameCPU f1(100, 100, {5, 5}, {15, 15});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 1.0f);
    f1.copyFrom(ptr);

    std::vector<Waypoint> path;
    for (int i = 2 * PATH_FEASIBLE_CPU_THRESHOLD; i >= 0; i--)
        path.push_back(Waypoint(50, i, angle::rad(0)));

    f1.processSafeDistanceZone({10, 10}, false);

    bool res = f1.checkFeasiblePath(path, 10, 10, true);

    ASSERT_FALSE(res);

    for (auto p : path)
    {
        if (p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCPUCheckFeasiblePath, CheckPath_Unfeasible_PreProcessWithVectorized_GPU)
{
    SearchFrameCPU f1(100, 100, {5, 5}, {15, 15});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 1.0f);
    f1.copyFrom(ptr);

    std::vector<Waypoint> path;
    for (int i = 2 * PATH_FEASIBLE_CPU_THRESHOLD; i >= 0; i--)
        path.push_back(Waypoint(50, i, angle::rad(0)));

    f1.processSafeDistanceZone({10, 10}, true);

    bool res = f1.checkFeasiblePath(path, 10, 10, true);

    ASSERT_FALSE(res);

    for (auto p : path)
    {
        if (p.is_checked_as_feasible())
            FAIL();
    }
}
