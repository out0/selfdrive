#include <gtest/gtest.h>
#include "../../include/search_frame.h"
#include "test_utils.h"
#include <cmath>

TEST(TestSearchFrame, TestSetGet)
{
    SearchFrame f1(1000, 1001, {50, 51}, {150, 151});

    ASSERT_EQ(1000, f1.width());
    ASSERT_EQ(1001, f1.height());

    // ASSERT_EQ(5, f1.minDistance().first);
    // ASSERT_EQ(6, f1.minDistance().second);

    ASSERT_EQ(50, f1.lowerBound().first);
    ASSERT_EQ(51, f1.lowerBound().second);

    ASSERT_EQ(150, f1.upperBound().first);
    ASSERT_EQ(151, f1.upperBound().second);
}

TEST(TestSearchFrame, TestConvertColors)
{
    SearchFrame f1(100, 100, {5, 5}, {15, 15});

    int c = 0;
    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
        {
            f1[{j, i}].x = c;
            c = (c + 1) % 29;
        }

    std::vector<std::tuple<int, int, int>> colors = {
        {0, 0, 0},
        {128, 64, 128},
        {244, 35, 232},
        {70, 70, 70},
        {102, 102, 156},
        {190, 153, 153},
        {153, 153, 153},
        {250, 170, 30},
        {220, 220, 0},
        {107, 142, 35},
        {152, 251, 152},
        {70, 130, 180},
        {220, 20, 60},
        {255, 0, 0},
        {0, 0, 142}, // car
        {0, 0, 70},
        {0, 60, 100},
        {0, 80, 100},
        {0, 0, 230},
        {119, 11, 32},
        {110, 190, 160},
        {170, 120, 50},
        {55, 90, 80},
        {45, 60, 150},
        {157, 234, 50},
        {81, 0, 81},
        {150, 100, 100},
        {230, 150, 140},
        {180, 165, 180}};

    f1.setClassColors(colors);

    uchar *dest = new uchar[3 * 100 * 100];
    f1.exportToColorFrame(dest);

    c = 0;
    int x, y, z;
    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++)
        {
            std::tie(x, y, z) = colors[c];
            int pos = 3 * (i * 100 + j);

            if (dest[pos] != x || dest[pos + 1] != y || dest[pos + 2] != z)
            {
                printf("error in position (%d, %d): expected (%d, %d, %d), obtained (%d, %d, %d)\n", i, j, x, y, z, dest[pos], dest[pos + 1], dest[pos + 2]);
                FAIL();
            }

            c = (c + 1) % 29;
        }
}

TEST(TestSearchFrame, TestClassCosts)
{
    SearchFrame f1(100, 100, {5, 5}, {15, 15});

    std::vector<float> costs({{0.0},
                              {1.0},
                              {2.0},
                              {3.1},
                              {4.2},
                              {5.3}});
    f1.setClassCosts(costs);

    ASSERT_DEQ(f1.getClassCost(0), 0.0);
    ASSERT_DEQ(f1.getClassCost(1), 1.0);
    ASSERT_DEQ(f1.getClassCost(2), 2.0);
    ASSERT_DEQ(f1.getClassCost(3), 3.1);
    ASSERT_DEQ(f1.getClassCost(4), 4.2);
    ASSERT_DEQ(f1.getClassCost(5), 5.3);
}

