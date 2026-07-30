#include <gtest/gtest.h>
#include "../../include/search_frame.h"
#include "test_utils.h"
#include <cmath>


TEST(TestSearchFrameSearchZone, TestPreProcessSearchZone)
{
    SearchFrame f1(100, 100, {5, 5}, {15, 15}, {10, 10});

    std::vector<float> costs({{0.0},
                              {-1.0}});
    f1.setClassCosts(costs);

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);
    f1.copyFrom(ptr);
    
    f1.processSafeDistanceZone({10, 10}, false);
}
