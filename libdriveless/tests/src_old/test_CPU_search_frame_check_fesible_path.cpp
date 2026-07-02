#include <gtest/gtest.h>
#include "../../include/search_frame_cpu.h"
#include "test_utils_cpu.h"
#include <cmath>

SearchFrameCPU *basicFrame()
{
    SearchFrameCPU *f1 = new SearchFrameCPU(100, 100, {-1, -1}, {-1, -1}, 1);
    f1->setClassCosts({{0.0}, {-1.0}, {0.0}, {0.0}});
    f1->setClassColors({{0, 0, 0}, {255, 255, 255}, {0, 255, 0}, {0, 0, 255}});
    return f1;
}

float *blankFrameData()
{
    const int SIZE = 3 * 100 * 100;
    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 0.0f);
    return ptr;
}

void toFrameData(float *ptr, std::vector<Waypoint> path)
{
    for (auto p : path)
        ptr[3 * (p.z() * 100 + p.x())] = 1.0;
}


TEST(TestSearchFrameCPUCheckFeasiblePath, CheckPath_Angle_NoPreProcess_CPU)
{
    SearchFrameCPU *frame = basicFrame();

    float *ptr = blankFrameData();
    toFrameData(ptr, testInterpolateHermiteCurve(100, 100, Waypoint(30, 99, angle::deg(0.0)), Waypoint(30, 0, angle::deg(0.0))));
    toFrameData(ptr, testInterpolateHermiteCurve(100, 100, Waypoint(70, 99, angle::deg(0.0)), Waypoint(70, 0, angle::deg(0.0))));

    frame->copyFrom(ptr);

    auto path1 = testInterpolateHermiteCurve(100, 100, Waypoint(50, 70, angle::deg(0.0)), Waypoint(50, 0, angle::deg(0.0)));
    //exportSearchFrameCPUToFile(*frame, "output1.png", path1);
    ASSERT_TRUE(frame->checkFeasiblePath(path1, 10, 10, true));
    ASSERT_TRUE(frame->checkFeasiblePath(path1, 10, 10, false));
    for (auto p : path1)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }

    auto path2 = testInterpolateHermiteCurve(100, 100, Waypoint(32, 70, angle::deg(0.0)), Waypoint(50, 0, angle::deg(0.0)));
    //exportSearchFrameCPUToFile(*frame, "output2.png", path2);
    ASSERT_FALSE(frame->checkFeasiblePath(path2, 10, 10, true));
    ASSERT_FALSE(frame->checkFeasiblePath(path2, 10, 10, false));  
}

TEST(TestSearchFrameCPUCheckFeasiblePath, CheckPath_Angle_PreProcessNoVectorized_CPU)
{
    SearchFrameCPU *frame = basicFrame();

    float *ptr = blankFrameData();
    toFrameData(ptr, testInterpolateHermiteCurve(100, 100, Waypoint(30, 99, angle::deg(0.0)), Waypoint(30, 0, angle::deg(0.0))));
    toFrameData(ptr, testInterpolateHermiteCurve(100, 100, Waypoint(70, 99, angle::deg(0.0)), Waypoint(70, 0, angle::deg(0.0))));

    frame->copyFrom(ptr);
    frame->processSafeDistanceZone({10, 10}, false);


    auto path1 = testInterpolateHermiteCurve(100, 100, Waypoint(50, 70, angle::deg(0.0)), Waypoint(50, 0, angle::deg(0.0)));
    //exportSearchFrameCPUToFile(*frame, "output1.png", path1);
    ASSERT_TRUE(frame->checkFeasiblePath(path1, 10, 10, true));
    ASSERT_TRUE(frame->checkFeasiblePath(path1, 10, 10, false));
    for (auto p : path1)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }

    auto path2 = testInterpolateHermiteCurve(100, 100, Waypoint(32, 70, angle::deg(0.0)), Waypoint(50, 0, angle::deg(0.0)));
    //exportSearchFrameCPUToFile(*frame, "output2.png", path2);
    ASSERT_FALSE(frame->checkFeasiblePath(path2, 10, 10, true));
    ASSERT_FALSE(frame->checkFeasiblePath(path2, 10, 10, false));

    
   
}

TEST(TestSearchFrameCPUCheckFeasiblePath, CheckPath_Angle_PreProcessWithVectorized_CPU)
{
    SearchFrameCPU *frame = basicFrame();

    float *ptr = blankFrameData();
    toFrameData(ptr, testInterpolateHermiteCurve(100, 100, Waypoint(30, 99, angle::deg(0.0)), Waypoint(30, 0, angle::deg(0.0))));
    toFrameData(ptr, testInterpolateHermiteCurve(100, 100, Waypoint(70, 99, angle::deg(0.0)), Waypoint(70, 0, angle::deg(0.0))));

    frame->copyFrom(ptr);
    frame->processSafeDistanceZone({10, 10}, true);


    auto path1 = testInterpolateHermiteCurve(100, 100, Waypoint(50, 70, angle::deg(0.0)), Waypoint(50, 0, angle::deg(0.0)));
    //exportSearchFrameCPUToFile(*frame, "output1.png", path1);
    ASSERT_TRUE(frame->checkFeasiblePath(path1, 10, 10, true));
    ASSERT_TRUE(frame->checkFeasiblePath(path1, 10, 10, false));
    for (auto p : path1)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }

    auto path2 = testInterpolateHermiteCurve(100, 100, Waypoint(32, 70, angle::deg(0.0)), Waypoint(50, 0, angle::deg(0.0)));
    //exportSearchFrameCPUToFile(*frame, "output2.png", path2);
    ASSERT_FALSE(frame->checkFeasiblePath(path2, 10, 10, true));
    ASSERT_FALSE(frame->checkFeasiblePath(path2, 10, 10, false));
}

TEST(TestSearchFrameCPUCheckFeasiblePath, CheckPath_Curved)
{
    SearchFrameCPU *frame = basicFrame();

    float *ptr = blankFrameData();
    toFrameData(ptr, testInterpolateHermiteCurve(100, 100, Waypoint(30, 99, angle::deg(0.0)), Waypoint(10, 0, angle::deg(0.0))));
    toFrameData(ptr, testInterpolateHermiteCurve(100, 100, Waypoint(70, 99, angle::deg(0.0)), Waypoint(50, 0, angle::deg(0.0))));

    frame->copyFrom(ptr);

    auto path1 = testInterpolateHermiteCurve(100, 100, Waypoint(50, 70, angle::deg(0.0)), Waypoint(30, 0, angle::deg(0.0)));
    exportSearchFrameCPUToFile(*frame, "output1.png", path1);
    ASSERT_TRUE(frame->checkFeasiblePath(path1, 10, 10, true));
    ASSERT_TRUE(frame->checkFeasiblePath(path1, 10, 10, false));
    for (auto p : path1)
    {
        if (!p.is_checked_as_feasible())
            FAIL();
    }

    auto path2 = testInterpolateHermiteCurve(100, 100, Waypoint(32, 70, angle::deg(0.0)), Waypoint(50, 0, angle::deg(0.0)));
    exportSearchFrameCPUToFile(*frame, "output2.png", path2);
    ASSERT_FALSE(frame->checkFeasiblePath(path2, 10, 10, true));
    ASSERT_FALSE(frame->checkFeasiblePath(path2, 10, 10, false));  
}
