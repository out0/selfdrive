#include <gtest/gtest.h>
#include "../../include/search_frame.h"
#include "../../include/angle.h"
#include "test_utils.h"
#include <cmath>


TEST(TestSearchFrameIsTraversable, TestTraversable)
{
    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    const int SIZE =  100 * 100 * 3;
    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 1.0f);

    f1.copyFrom(ptr);
    f1.setClassCosts({-1, 0.0});

    bool res = f1.isTraversable(0, 0, angle::deg(0), true);
    ASSERT_TRUE(res);
    
    res = f1.isTraversable(0, 0, angle::deg(0), false);
    ASSERT_TRUE(res);

    f1.processSafeDistanceZone({10, 10}, false);
    res = f1.isTraversable(0, 0, angle::deg(0), true);
    ASSERT_TRUE(res);
    res = f1.isTraversable(0, 0, angle::deg(0), false);
    ASSERT_TRUE(res);

    f1.processSafeDistanceZone({10, 10}, true);
    res = f1.isTraversable(0, 0, angle::deg(0), true);
    ASSERT_TRUE(res);
    res = f1.isTraversable(0, 0, angle::deg(0), false);
    ASSERT_TRUE(res);
    
}

TEST(TestSearchFrameIsTraversable, TestNotTraversable)
{
    SearchFrame f1(100, 100, {-1, -1}, {-1, -1});

    const int SIZE =  100 * 100 * 3;
    float *ptr = new float[SIZE];
    std::fill(ptr, ptr + SIZE, 1.0f);

    f1.copyFrom(ptr);
    f1.setClassCosts({-1, -1});

    bool res = f1.isTraversable(0, 0, angle::deg(0), true);

    ASSERT_FALSE(res);
    
    res = f1.isTraversable(0, 0, angle::deg(0), false);
    ASSERT_FALSE(res);

    f1.processSafeDistanceZone({10, 10}, false);
    res = f1.isTraversable(0, 0, angle::deg(0), true);
    ASSERT_FALSE(res);
    res = f1.isTraversable(0, 0, angle::deg(0), false);
    ASSERT_FALSE(res);

    f1.processSafeDistanceZone({10, 10}, true);
    res = f1.isTraversable(0, 0, angle::deg(0), true);
    ASSERT_FALSE(res);
    res = f1.isTraversable(0, 0, angle::deg(0), false);
    ASSERT_FALSE(res);
    
}

