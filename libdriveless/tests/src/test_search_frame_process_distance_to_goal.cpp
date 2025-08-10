#include <gtest/gtest.h>
#include "../../include/search_frame.h"
#include "test_utils.h"
#include <cmath>


TEST(TestSearchFrameProcessDistanceToGoal, ProcDistance)
{
    SearchFrame f1(100, 100, {5, 5}, {15, 15});

    const int SIZE = 3 * 100 * 100;

    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);
    f1.copyFrom(ptr);

    f1.processDistanceToGoal(50, -100);

    for (int z = 0; z < 100; z++)
        for (int x = 0; x < 100; x++) {
            float dist = f1.getDistanceToGoal(x, z);
            float dx = x - 50;
            float dz = z + 100;
            float d = sqrt(dx * dx + dz * dz);
            if (std::abs(d - dist) > 0.01)
                FAIL();
        }
            
}

