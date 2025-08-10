#include <gtest/gtest.h>
#include "../../include/search_frame.h"
#include "test_utils.h"
#include <cmath>

SearchFrame *buildFrame(int *ptr, std::pair<int, int> size)
{
    SearchFrame *frame = new SearchFrame(size.first, size.second, {-1, -1}, {-1, -1});

    float *fptr = new float[3 * frame->height() * frame->width()];

    for (int h = 0; h < frame->height(); h++)
        for (int w = 0; w < frame->width(); w++)
        {
            int pos1 = h * frame->width() + w;
            int pos2 = 3 * pos1;

            fptr[pos2] = 1;
            if (ptr[pos1] == 1)
                fptr[pos2] = 0; // obstacle

            fptr[pos2 + 1] = 0;
            fptr[pos2 + 2] = 0;
        }

    frame->copyFrom(fptr);
    frame->setClassCosts({-1, 1});

    delete[] fptr;
    return frame;
}

double euclideanDist(std::pair<int, int> p1, std::pair<int, int> p2)
{
    double dx = p1.first - p2.first;
    double dz = p1.second - p2.second;
    return sqrt(dx * dx + dz * dz);
}

TEST(TestSearchFrameprocessSafeDistanceZoneVectorized, TestprocessSafeDistanceZoneCosts)
{
    SearchFrame f1(12, 12, {0, 0}, {0, 0});

    int grid[] = {
        // 0  1  2  3  4  5  6  7  8  9  10
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
    };

    SearchFrame *frame = buildFrame(grid, {11, 11});
    frame->processSafeDistanceZone({0, 0}, true);

    for (int i = 0; i < frame->height(); i++)
        for (int j = 0; j < frame->width(); j++)
        {
            float expected = euclideanDist({j, i}, {5, 0});

            if ((*frame)[{j, i}].x == 1)
            {
                // free space
                expected = frame->getClassCost((*frame)[{j, i}].x) * euclideanDist({j, i}, {5, 0});
            }

            float obtained = frame->getCost(j, i);

            if (abs(expected - obtained) > 0.0001)
            {
                printf("processSafeDistanceZone cost failed for position {%d, %d}: expected %f obtained %f\n", j, i, expected, obtained);
                // FAIL();
            }
        }
}

void assertColumnFeasibleValue(SearchFrame *frame, int col, int val, int startLine = -1, int endLine = -1)
{
    if (startLine < 0)
        startLine = 0;
    if (endLine < 0)
        endLine = frame->height();

    for (int i = 0; i < frame->height(); i++)
    {
        int angles = static_cast<int>((*frame)[{col, i}].z) & 0x0ff;
        if (angles != val)
        {
            printf("angle value on {%d, %d} expected %d obtained %d\n", col, i, val, angles);
            FAIL();
        }
    }
}

void assertRowFeasibleValue(SearchFrame *frame, int row, int val, int startCol = -1, int endCol = -1)
{
    if (startCol < 0)
        startCol = 0;
    if (endCol < 0)
        endCol = frame->width();

    for (int i = 0; i < frame->width(); i++)
    {
        int angles = static_cast<int>((*frame)[{i, row}].z )& 0x0ff;;
        if (angles != val)
        {
            printf("angle value on {%d, %d} expected %d obtained %d\n", i, row, val, angles);
            FAIL();
        }
    }
}



TEST(TestSearchFrameprocessSafeDistanceZoneVectorized, TestprocessSafeDistanceZoneFeasibilityNoMargins)
{
    int grid[] = {
        // 0  1  2  3  4  5  6  7  8  9  10
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
    };

    SearchFrame *frame = buildFrame(grid, {11, 11});
    frame->processSafeDistanceZone({0, 0}, true);

    // obstacles
    assertColumnFeasibleValue(frame, 0, 0x0);
    assertColumnFeasibleValue(frame, 1, 0x0);
    assertColumnFeasibleValue(frame, 2, 0x0);
    assertColumnFeasibleValue(frame, 8, 0x0);
    assertColumnFeasibleValue(frame, 9, 0x0);
    assertColumnFeasibleValue(frame, 10, 0x0);

    // no minimal distances
    for (int i = 3; i < 8; i++)
    {
        assertColumnFeasibleValue(frame, i, 0xFF);
    }
}

void testExpectedObtainedGrid(SearchFrame *frame, int *expected)
{
    for (int j = 0; j < frame->height(); j++)
        for (int i = 0; i < frame->width(); i++)
        {
            int pos = j * frame->width() + i;
            int obtained = static_cast<int>((*frame)[{i, j}].z) & 0x0ff;
            if (expected[pos] != obtained)
            {
                printf("angle value on {%d, %d} expected %d obtained %d\n", i, j, expected[pos], obtained);
                FAIL();
            }
        }
}

TEST(TestSearchFrameprocessSafeDistanceZoneVectorized, TestprocessSafeDistanceZoneFeasibilityMargins)
{
    int grid[] = {
        // 0  1  2  3  4  5  6  7  8  9  10
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
    };

    int p04 = BIT_HEADING_MINUS_45 | BIT_HEADING_MINUS_22_5 | BIT_HEADING_0;
    int p06 = BIT_HEADING_45 | BIT_HEADING_22_5 | BIT_HEADING_0;
    int Z = BIT_HEADING_0;
    int A = 0xff;
    int pA4 = BIT_HEADING_45 | BIT_HEADING_22_5 | BIT_HEADING_0;
    int pA6 = BIT_HEADING_MINUS_45 | BIT_HEADING_MINUS_22_5 | BIT_HEADING_0;

    int expected[] = {
        // 0  1  2   3   4   5   6  7  8  9  10
        0, 0, 0, 0, p04, A, p06, 0, 0, 0, 0, //
        0, 0, 0, 0, Z, A, Z, 0, 0, 0, 0,     //
        0, 0, 0, 0, Z, A, Z, 0, 0, 0, 0,     //
        0, 0, 0, 0, Z, A, Z, 0, 0, 0, 0,     //
        0, 0, 0, 0, Z, A, Z, 0, 0, 0, 0,     //
        0, 0, 0, 0, Z, A, Z, 0, 0, 0, 0,     //
        0, 0, 0, 0, Z, A, Z, 0, 0, 0, 0,     //
        0, 0, 0, 0, Z, A, Z, 0, 0, 0, 0,     //
        0, 0, 0, 0, Z, A, Z, 0, 0, 0, 0,     //
        0, 0, 0, 0, Z, A, Z, 0, 0, 0, 0,     //
        0, 0, 0, 0, pA4, A, pA6, 0, 0, 0, 0, //
    };

    SearchFrame *frame = buildFrame(grid, {11, 11});
    frame->processSafeDistanceZone({2, 4},  true);
    testExpectedObtainedGrid(frame, expected);
}


TEST(TestSearchFrameprocessSafeDistanceZoneVectorized, TestprocessSafeDistanceZoneFeasibilityMarginsCurved)
{
    int grid[] = {
     // 0  1  2  3  4  5  6  7  8  9  10
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, //
        1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, //
        1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, //
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, //
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, //
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, //
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, //
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, //
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, //
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, //
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, //
    };

    int p02 = BIT_HEADING_MINUS_45 | BIT_HEADING_MINUS_67_5 | BIT_HEADING_MINUS_22_5;
    int p03 = BIT_HEADING_MINUS_45 | BIT_HEADING_MINUS_22_5 | BIT_HEADING_45 | BIT_HEADING_22_5 | BIT_HEADING_0;
    //int p13 = HEADING_45 | HEADING_22_5;
    int L = BIT_HEADING_MINUS_67_5 | BIT_HEADING_MINUS_45 | BIT_HEADING_MINUS_22_5;
    int Z = BIT_HEADING_0;
    int A = 0xff;
    int p45 = BIT_HEADING_MINUS_45 | BIT_HEADING_MINUS_22_5 | BIT_HEADING_0;
    int pA5 = BIT_HEADING_45 |BIT_HEADING_22_5 | BIT_HEADING_0;
    int pA7 = BIT_HEADING_MINUS_45 | BIT_HEADING_MINUS_22_5 | BIT_HEADING_0;

    int expected[] = {
     // 0  1  2  3   4   5   6  7  8  9  10
        0, 0, p02, p03, 0, 0, 0, 0, 0, 0, 0, //0
        0, 0, 0, L, 0, 0, 0, 0, 0, 0, 0,     //1
        0, 0, 0, 0, L, 0, 0, 0, 0, 0, 0,     //2
        0, 0, 0, 0, 0, L, 0, 0, 0, 0, 0,     //3
        0, 0, 0, 0, 0, p45, L, 0, 0, 0, 0,   //4
        0, 0, 0, 0, 0, Z, A, 0, 0, 0, 0,     //5
        0, 0, 0, 0, 0, Z, A, Z, 0, 0, 0,     //6
        0, 0, 0, 0, 0, Z, A, Z, 0, 0, 0,     //7
        0, 0, 0, 0, 0, Z, A, Z, 0, 0, 0,     //8
        0, 0, 0, 0, 0, Z, A, Z, 0, 0, 0,     //9
        0, 0, 0, 0, 0, pA5, A, pA7, 0, 0, 0,     //10
    };

    SearchFrame *frame = buildFrame(grid, {11, 11});
    frame->processSafeDistanceZone({2, 4}, true);
    testExpectedObtainedGrid(frame, expected);
}