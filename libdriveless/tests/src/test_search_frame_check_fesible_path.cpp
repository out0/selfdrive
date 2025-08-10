#include <gtest/gtest.h>
#include "../../include/search_frame.h"
#include "test_utils.h"
#include <cmath>

// CURVED
// ---------------------------------------------------------------

TEST(TestSearchFrameCheckFeasiblePath, CheckPath_Angle_NoPreProcess_CPU)
{
    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0},
                              {0.0}});
    f1.setClassCosts(costs);
    f1.setClassColors({{0, 0, 0}, {255, 255, 255}, {0, 255, 0}});

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 0.0f);

    auto c1 = testInterpolateHermiteCurve(100, 100, Waypoint(30, 99, angle::deg(0.0)), Waypoint(30, 0, angle::deg(0.0)));
    for (auto p : c1)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;
    auto c2 = testInterpolateHermiteCurve(100, 100, Waypoint(70, 99, angle::deg(0.0)), Waypoint(70, 0, angle::deg(0.0)));
    for (auto p : c2)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;

    // auto c3 = testInterpolateHermiteCurve(100, 100, Waypoint(36, 99, angle::deg(0.0)), Waypoint(36, 0, angle::deg(0.0)));
    // for (auto p : c3)
    //     ptr[3 * (p.z() * 100 + p.x())] = 2.0;

    f1.copyFrom(ptr);

    // exportSearchFrameToFile(f1, "output.png");

    std::vector<Waypoint> path1;
    for (int i = (PATH_FEASIBLE_CPU_THRESHOLD - 1); i >= 0; i--)
        path1.push_back(Waypoint(34, i, angle::rad(0)));

    bool res = f1.checkFeasiblePath(path1, 10, 10, true);

    std::vector<Waypoint> path2;
    for (int i = (PATH_FEASIBLE_CPU_THRESHOLD - 1); i >= 0; i--)
        path2.push_back(Waypoint(36, i, angle::rad(0)));

    bool res2 = f1.checkFeasiblePath(path2, 10, 10, true);

    ASSERT_FALSE(res);

    for (auto p : path1)
    {
        if (p.is_checked_as_feasible())
            FAIL();
    }

    ASSERT_TRUE(res2);

    for (auto p : path2)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCheckFeasiblePath, CheckPath_Angle_PreProcessNoVectorized_CPU)
{
    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 0.0f);

    auto c1 = testInterpolateHermiteCurve(100, 100, Waypoint(30, 99, angle::deg(0.0)), Waypoint(30, 0, angle::deg(0.0)));
    for (auto p : c1)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;
    auto c2 = testInterpolateHermiteCurve(100, 100, Waypoint(70, 99, angle::deg(0.0)), Waypoint(70, 0, angle::deg(0.0)));
    for (auto p : c2)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;

    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, false);

    std::vector<Waypoint> path1;
    for (int i = (PATH_FEASIBLE_CPU_THRESHOLD - 1); i >= 0; i--)
        path1.push_back(Waypoint(34, i, angle::rad(0)));

    bool res = f1.checkFeasiblePath(path1, 10, 10, true);

    std::vector<Waypoint> path2;
    for (int i = (PATH_FEASIBLE_CPU_THRESHOLD - 1); i >= 0; i--)
        path2.push_back(Waypoint(36, i, angle::rad(0)));

    bool res2 = f1.checkFeasiblePath(path2, 10, 10, true);

    ASSERT_FALSE(res);

    for (auto p : path1)
    {
        if (p.is_checked_as_feasible())
            FAIL();
    }

    ASSERT_TRUE(res2);

    for (auto p : path2)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCheckFeasiblePath, CheckPath_Angle_PreProcessWithVectorized_CPU)
{
    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 0.0f);

    auto c1 = testInterpolateHermiteCurve(100, 100, Waypoint(30, 99, angle::deg(0.0)), Waypoint(30, 0, angle::deg(0.0)));
    for (auto p : c1)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;
    auto c2 = testInterpolateHermiteCurve(100, 100, Waypoint(70, 99, angle::deg(0.0)), Waypoint(70, 0, angle::deg(0.0)));
    for (auto p : c2)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;

    f1.copyFrom(ptr);

    // exportSearchFrameToFile(f1, "output.png");
    f1.processSafeDistanceZone({10, 10}, true);

    std::vector<Waypoint> path1;
    for (int i = (PATH_FEASIBLE_CPU_THRESHOLD - 1); i >= 0; i--)
        path1.push_back(Waypoint(34, i, angle::rad(0)));

    bool res = f1.checkFeasiblePath(path1, 10, 10, true);

    std::vector<Waypoint> path2;
    for (int i = (PATH_FEASIBLE_CPU_THRESHOLD - 1); i >= 0; i--)
        path2.push_back(Waypoint(36, i, angle::rad(0)));

    bool res2 = f1.checkFeasiblePath(path2, 10, 10, true);

    ASSERT_FALSE(res);

    for (auto p : path1)
    {
        if (p.is_checked_as_feasible())
            FAIL();
    }

    ASSERT_TRUE(res2);

    for (auto p : path2)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCheckFeasiblePath, CheckPath_Angle_NoPreProcess_GPU)
{
    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);
    f1.setClassColors({{0, 0, 0}, {255, 255, 255}});

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 0.0f);

    auto c1 = testInterpolateHermiteCurve(100, 100, Waypoint(30, 99, angle::deg(0.0)), Waypoint(30, 0, angle::deg(0.0)));
    for (auto p : c1)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;
    auto c2 = testInterpolateHermiteCurve(100, 100, Waypoint(70, 99, angle::deg(0.0)), Waypoint(70, 0, angle::deg(0.0)));
    for (auto p : c2)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;

    f1.copyFrom(ptr);

    std::vector<Waypoint> path1;
    for (int i = 2 * PATH_FEASIBLE_CPU_THRESHOLD; i >= 0; i--)
        path1.push_back(Waypoint(34, i, angle::rad(0)));

    bool res = f1.checkFeasiblePath(path1, 10, 10, true);

    std::vector<Waypoint> path2;
    for (int i = 2 * PATH_FEASIBLE_CPU_THRESHOLD; i >= 0; i--)
        path2.push_back(Waypoint(36, i, angle::rad(0)));

    bool res2 = f1.checkFeasiblePath(path2, 10, 10, true);

    ASSERT_FALSE(res);

    for (auto p : path1)
    {
        if (p.is_checked_as_feasible())
            FAIL();
    }

    ASSERT_TRUE(res2);

    for (auto p : path2)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCheckFeasiblePath, CheckPath_Angle_PreProcessNoVectorized_GPU)
{
    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);
    f1.setClassColors({{0, 0, 0}, {255, 255, 255}});

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 0.0f);

    auto c1 = testInterpolateHermiteCurve(100, 100, Waypoint(30, 99, angle::deg(0.0)), Waypoint(30, 0, angle::deg(0.0)));
    for (auto p : c1)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;
    auto c2 = testInterpolateHermiteCurve(100, 100, Waypoint(70, 99, angle::deg(0.0)), Waypoint(70, 0, angle::deg(0.0)));
    for (auto p : c2)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;

    f1.copyFrom(ptr);
    f1.processSafeDistanceZone({10, 10}, false);

    std::vector<Waypoint> path1;
    for (int i = 2 * PATH_FEASIBLE_CPU_THRESHOLD; i >= 0; i--)
        path1.push_back(Waypoint(34, i, angle::rad(0)));

    bool res = f1.checkFeasiblePath(path1, 10, 10, true);

    std::vector<Waypoint> path2;
    for (int i = 2 * PATH_FEASIBLE_CPU_THRESHOLD; i >= 0; i--)
        path2.push_back(Waypoint(36, i, angle::rad(0)));

    bool res2 = f1.checkFeasiblePath(path2, 10, 10, true);

    ASSERT_FALSE(res);

    for (auto p : path1)
    {
        if (p.is_checked_as_feasible())
            FAIL();
    }

    ASSERT_TRUE(res2);

    for (auto p : path2)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }
}

TEST(TestSearchFrameCheckFeasiblePath, CheckPath_Angle_PreProcessWithVectorized_GPU)
{
    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);
    f1.setClassColors({{0, 0, 0}, {255, 255, 255}});

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 0.0f);

    auto c1 = testInterpolateHermiteCurve(100, 100, Waypoint(30, 99, angle::deg(0.0)), Waypoint(30, 0, angle::deg(0.0)));
    for (auto p : c1)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;
    auto c2 = testInterpolateHermiteCurve(100, 100, Waypoint(70, 99, angle::deg(0.0)), Waypoint(70, 0, angle::deg(0.0)));
    for (auto p : c2)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;

    f1.copyFrom(ptr);
    f1.processSafeDistanceZone({10, 10}, true);

    std::vector<Waypoint> path1;
    for (int i = 2 * PATH_FEASIBLE_CPU_THRESHOLD; i >= 0; i--)
        path1.push_back(Waypoint(34, i, angle::rad(0)));

    bool res = f1.checkFeasiblePath(path1, 10, 10, true);

    std::vector<Waypoint> path2;
    for (int i = 2 * PATH_FEASIBLE_CPU_THRESHOLD; i >= 0; i--)
        path2.push_back(Waypoint(36, i, angle::rad(0)));

    bool res2 = f1.checkFeasiblePath(path2, 10, 10, true);

    ASSERT_FALSE(res);

    for (auto p : path1)
    {
        if (p.is_checked_as_feasible())
            FAIL();
    }

    ASSERT_TRUE(res2);

    for (auto p : path2)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }
}
