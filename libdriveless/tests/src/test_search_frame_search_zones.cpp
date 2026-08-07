#include <gtest/gtest.h>
#include "../../include/search_frame.h"
#include "test_utils.h"
#include <cmath>

#include <vector>

namespace
{
    // Frame geometry shared by every test below.
    constexpr int FRAME_W = 100;
    constexpr int FRAME_H = 100;
    constexpr int ZONE_W = 10;
    constexpr int ZONE_H = 10;

    // Ego bounds intentionally kept identical to the existing
    // TestPreProcessSearchZone test above (this particular combination
    // makes the "ignore ego rectangle" check a no-op, so it never
    // interferes with obstacle counting in these tests).
    const std::pair<int, int> LOWER_BOUND = {5, 5};
    const std::pair<int, int> UPPER_BOUND = {15, 15};

    // Builds a flat float buffer representing the frame (class, cost,
    // traversability per pixel), defaulting every pixel to class 0
    // (non-obstacle, cost 0.0).
    std::vector<float> buildBlankFrameBuffer()
    {
        std::vector<float> buf(3 * FRAME_W * FRAME_H, 0.0f);
        return buf;
    }

    // Marks pixel (x, z) as belonging to the obstacle class (class id 1,
    // whose cost is set to -1.0 by setupObstacleClassCosts()).
    void setObstaclePixel(std::vector<float> &buf, int x, int z)
    {
        const long pos = (static_cast<long>(z) * FRAME_W + x) * 3;
        buf[pos] = 1.0f; // segmentation class id 1 == obstacle
    }

    void setupObstacleClassCosts(SearchFrame &f)
    {
        // class 0 => traversable (cost 0.0), class 1 => obstacle (cost < 0)
        f.setClassCosts({0.0f, -1.0f});
    }
}

TEST(TestSearchFrame_SearchZone, TestZoneDivision_Width)
{
    SearchFrame f1(50, 50, {-1, -1}, {-1, -1}, {10, 10});

    for (int h = 0; h < 50; h++)
    {
        // Check width division
        for (int w = 0; w < 50; w++)
        {
            int2 p = f1.getSearchZoneLocation(w, h);
            if (w >= 0 && w < 10)
                ASSERT_EQ(p.x, 0);
            else if (w >= 10 && w < 20)
                ASSERT_EQ(p.x, 1);
            else if (w >= 20 && w < 30)
                ASSERT_EQ(p.x, 2);
            else if (w >= 30 && w < 40)
                ASSERT_EQ(p.x, 3);
            else if (w >= 40 && w < 50)
                ASSERT_EQ(p.x, 4);
            else if (w >= 50)
                ASSERT_EQ(p.x, -1);
        }
    }
}
TEST(TestSearchFrame_SearchZone, TestZoneDivision_Height)
{
    SearchFrame f1(50, 50, {-1, -1}, {-1, -1}, {10, 10});

    for (int w = 0; w < 50; w++)
    {
        // Check height division
        for (int h = 0; h < 50; h++)
        {
            int2 p = f1.getSearchZoneLocation(w, h);
            if (h >= 0 && h < 10)
                ASSERT_EQ(p.y, 0);
            else if (h >= 10 && h < 20)
                ASSERT_EQ(p.y, 1);
            else if (h >= 20 && h < 30)
                ASSERT_EQ(p.y, 2);
            else if (h >= 30 && h < 40)
                ASSERT_EQ(p.y, 3);
            else if (h >= 40 && h < 50)
                ASSERT_EQ(p.y, 4);
            else if (h >= 50)
                ASSERT_EQ(p.y, -1);
        }
    }
}
TEST(TestSearchFrame_SearchZone, TestZoneDivision_Both)
{
    SearchFrame f1(50, 50, {-1, -1}, {-1, -1}, {10, 10});

    for (int w = 0; w < 50; w++)
    {
        // Check height division
        for (int h = 0; h < 50; h++)
        {
            int2 p = f1.getSearchZoneLocation(w, h);

            if (w >= 0 && w < 10)
                ASSERT_EQ(p.x, 0);
            else if (w >= 10 && w < 20)
                ASSERT_EQ(p.x, 1);
            else if (w >= 20 && w < 30)
                ASSERT_EQ(p.x, 2);
            else if (w >= 30 && w < 40)
                ASSERT_EQ(p.x, 3);
            else if (w >= 40 && w < 50)
                ASSERT_EQ(p.x, 4);
            else if (w >= 50)
                ASSERT_EQ(p.x, -1);

            if (h >= 0 && h < 10)
                ASSERT_EQ(p.y, 0);
            else if (h >= 10 && h < 20)
                ASSERT_EQ(p.y, 1);
            else if (h >= 20 && h < 30)
                ASSERT_EQ(p.y, 2);
            else if (h >= 30 && h < 40)
                ASSERT_EQ(p.y, 3);
            else if (h >= 40 && h < 50)
                ASSERT_EQ(p.y, 4);
            else if (h >= 50)
                ASSERT_EQ(p.y, -1);
        }
    }
}

TEST(TestSearchFrame_SearchZone, TestZoneLocation)
{
    int2 matrix[10][10] = {
        {{0, 0}, {0, 0}, {1, 0}, {1, 0}, {2, 0}, {2, 0}, {3, 0}, {3, 0}, {4, 0}, {4, 0}},
        {{0, 0}, {0, 0}, {1, 0}, {1, 0}, {2, 0}, {2, 0}, {3, 0}, {3, 0}, {4, 0}, {4, 0}},
        {{0, 1}, {0, 1}, {1, 1}, {1, 1}, {2, 1}, {2, 1}, {3, 1}, {3, 1}, {4, 1}, {4, 1}},
        {{0, 1}, {0, 1}, {1, 1}, {1, 1}, {2, 1}, {2, 1}, {3, 1}, {3, 1}, {4, 1}, {4, 1}},
        {{0, 2}, {0, 2}, {1, 2}, {1, 2}, {2, 2}, {2, 2}, {3, 2}, {3, 2}, {4, 2}, {4, 2}},
        {{0, 2}, {0, 2}, {1, 2}, {1, 2}, {2, 2}, {2, 2}, {3, 2}, {3, 2}, {4, 2}, {4, 2}},
        {{0, 3}, {0, 3}, {1, 3}, {1, 3}, {2, 3}, {2, 3}, {3, 3}, {3, 3}, {4, 3}, {4, 3}},
        {{0, 3}, {0, 3}, {1, 3}, {1, 3}, {2, 3}, {2, 3}, {3, 3}, {3, 3}, {4, 3}, {4, 3}},
        {{0, 4}, {0, 4}, {1, 4}, {1, 4}, {2, 4}, {2, 4}, {3, 4}, {3, 4}, {4, 4}, {4, 4}},
        {{0, 4}, {0, 4}, {1, 4}, {1, 4}, {2, 4}, {2, 4}, {3, 4}, {3, 4}, {4, 4}, {4, 4}},
    };

    SearchFrame f1(10, 10, {-1, -1}, {-1, -1}, {2, 2});

    for (int w = 0; w < 10; w++)
    {
        for (int h = 0; h < 10; h++)
        {
            int2 expected_location = matrix[h][w];
            int2 obtained_location = f1.getSearchZoneLocation(w, h);

            if (expected_location.x != obtained_location.x || expected_location.y != obtained_location.y)
            {
                printf("failed for addr: %d, %d - expected: %d,%d obtained: %d,%d\n", w, h, expected_location.x, expected_location.y, obtained_location.x, obtained_location.y);
                FAIL();
            }
        }
    }
}

TEST(TestSearchFrame_SearchZone, TestZoneDivision_Id)
{
    int matrix[10][10] = {
        {0, 0, 1, 1, 2, 2, 3, 3, 4, 4},
        {0, 0, 1, 1, 2, 2, 3, 3, 4, 4},
        {5, 5, 6, 6, 7, 7, 8, 8, 9, 9},
        {5, 5, 6, 6, 7, 7, 8, 8, 9, 9},
        {10, 10, 11, 11, 12, 12, 13, 13, 14, 14},
        {10, 10, 11, 11, 12, 12, 13, 13, 14, 14},
        {15, 15, 16, 16, 17, 17, 18, 18, 19, 19},
        {15, 15, 16, 16, 17, 17, 18, 18, 19, 19},
        {20, 20, 21, 21, 22, 22, 23, 23, 24, 24},
        {20, 20, 21, 21, 22, 22, 23, 23, 24, 24},
    };

    SearchFrame f1(10, 10, {-1, -1}, {-1, -1}, {2, 2});

    for (int w = 0; w < 10; w++)
    {
        for (int h = 0; h < 10; h++)
        {
            int expected_id = matrix[h][w];
            if (expected_id != f1.getSearchZoneId(w, h))
            {
                printf("failed for addr: %d, %d - expected: %d obtained: %d\n", w, h, expected_id, f1.getSearchZoneId(w, h));
                int2 l = f1.getSearchZoneLocation(w, h);
                printf("search zone location for %d, %d is %d, %d\n", w, h, l.x, l.y);
            }
            ASSERT_EQ(expected_id, f1.getSearchZoneId(w, h));
        }
    }
}

// A freshly processed SearchFrame with no obstacles anywhere must report
// {0, 0} (no obstacles, no border obstacles) for every zone in the grid.
// (Not affected by the flat-index bug: there is nothing to miscount.)
TEST(TestSearchFrame_SearchZone, ReadSearchZoneInfo_NoObstacle_ReturnsZeroForEveryZone)
{
    SearchFrame f(FRAME_W, FRAME_H, LOWER_BOUND, UPPER_BOUND, {ZONE_W, ZONE_H});
    setupObstacleClassCosts(f);

    std::vector<float> buf = buildBlankFrameBuffer();
    f.copyFrom(buf.data());
    f.processSafeDistanceZone({ZONE_W, ZONE_H}, false);

    f.setClassColors({{0, 0, 0},
                      {255, 255, 255}});

    int2 gridSize = f.getSearchZoneGridSize();

    for (int zz = 0; zz < gridSize.y; zz++)
    {
        for (int xx = 0; xx < gridSize.x; xx++)
        {
            uint4 info = f.readSearchZoneInfo(xx, zz);
            EXPECT_EQ(info.x, 0u) << "zone (" << xx << "," << zz << ") obstacle count";
            EXPECT_EQ(info.y, 0u) << "zone (" << xx << "," << zz << ") border obstacle count";
        }
    }
}

TEST(TestSearchFrame_SearchZone, ReadSearchZoneInfo_ObstacleInOneZone)
{
    SearchFrame f(FRAME_W, FRAME_H, LOWER_BOUND, UPPER_BOUND, {ZONE_W, ZONE_H});
    setupObstacleClassCosts(f);

    std::vector<float> buf = buildBlankFrameBuffer();

    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            setObstaclePixel(buf, i, j);

    f.copyFrom(buf.data());
    f.processSafeDistanceZone({ZONE_W, ZONE_H}, false);

    f.setClassColors({{0, 0, 0},
                      {255, 255, 255}});
    
                      exportSearchFrameToFile(f, "output.png", true, true);

    int2 gridSize = f.getSearchZoneGridSize();

    uint4 info = f.readSearchZoneInfo(0, 0);
    EXPECT_EQ(info.x, 100);
    EXPECT_EQ(info.y, 36);

    for (int h = 0; h < gridSize.y; h++)
    {
        for (int w = 0; w < gridSize.x; w++)
        {
            uint4 info = f.readSearchZoneInfo(w, h);
            if (w == 0 && h == 0)
            {
                EXPECT_EQ(info.x, 100);
                EXPECT_EQ(info.y, 36);
            }
            else
            {
                EXPECT_EQ(info.x, 0);
                EXPECT_EQ(info.y, 0);
            }
        }
    }
}

// A single obstacle placed exactly on the outer edge of zone (3,1) should
// increment *both* the obstacle counter (.x) and the border counter (.y)
// when read back from THAT zone.
// EXPECTED TO FAIL against the current implementation, for the same
// flat-index reason as above.
TEST(TestSearchFrameSearchZone, ReadSearchZoneInfo_BorderObstacle_CountsAsObstacleAndBorder)
{
    SearchFrame f(FRAME_W, FRAME_H, LOWER_BOUND, UPPER_BOUND, {ZONE_W, ZONE_H});
    setupObstacleClassCosts(f);

    // Zone (xg=3, zg=1) covers pixels x in [30,39], z in [10,19].
    // (35, 10) sits on the top edge (z == zone start) of that zone.
    std::vector<float> buf = buildBlankFrameBuffer();
    setObstaclePixel(buf, 35, 10);
    f.copyFrom(buf.data());

    f.processSafeDistanceZone({ZONE_W, ZONE_H}, false);

    uint4 info = f.readSearchZoneInfo(3, 1);
    EXPECT_EQ(info.x, 1u) << "obstacle at pixel (35,10) should be counted in zone (3,1)";
    EXPECT_EQ(info.y, 1u) << "obstacle at pixel (35,10) sits on the border of zone (3,1)";
}

// // Multiple obstacles landing in the same zone (4,2) must accumulate: an
// // interior obstacle plus a border obstacle should produce an obstacle
// // count of 2 and a border count of 1, when read back from THAT zone.
// // EXPECTED TO FAIL against the current implementation (zg=2 also hits
// // the flat-index bug).
TEST(TestSearchFrameSearchZone, ReadSearchZoneInfo_MultipleObstaclesInSameZone_CountsAccumulate)
{
    SearchFrame f(FRAME_W, FRAME_H, LOWER_BOUND, UPPER_BOUND, {ZONE_W, ZONE_H});
    setupObstacleClassCosts(f);

    // Zone (xg=4, zg=2) covers pixels x in [40,49], z in [20,29].
    std::vector<float> buf = buildBlankFrameBuffer();
    setObstaclePixel(buf, 45, 25); // interior obstacle
    setObstaclePixel(buf, 40, 25); // border obstacle (x == zone start)
    f.copyFrom(buf.data());

    f.processSafeDistanceZone({ZONE_W, ZONE_H}, false);

    uint4 info = f.readSearchZoneInfo(4, 2);
    EXPECT_EQ(info.x, 2u);
    EXPECT_EQ(info.y, 1u);
}

// // Obstacles in different, non-adjacent zones must be counted
// // independently: each populated zone reflects only its own obstacles,
// // and every other zone (including ones in between) remains {0, 0}.
// // zoneA sits in row zg=0 (unaffected by the flat-index bug and expected
// // to pass); zoneB sits in row zg=4 and is EXPECTED TO FAIL for the same
// // reason as the tests above.
TEST(TestSearchFrameSearchZone, ReadSearchZoneInfo_ObstaclesInDifferentZones_AreIndependent)
{
    SearchFrame f(FRAME_W, FRAME_H, LOWER_BOUND, UPPER_BOUND, {ZONE_W, ZONE_H});
    setupObstacleClassCosts(f);

    std::vector<float> buf = buildBlankFrameBuffer();
    setObstaclePixel(buf, 25, 5);  // zone (2, 0): interior obstacle
    setObstaclePixel(buf, 65, 45); // zone (6, 4): interior obstacle
    f.copyFrom(buf.data());

    f.processSafeDistanceZone({ZONE_W, ZONE_H}, false);

    uint4 zoneWithObstacleA = f.readSearchZoneInfo(2, 0);
    uint4 zoneWithObstacleB = f.readSearchZoneInfo(6, 4);
    uint4 untouchedZone = f.readSearchZoneInfo(3, 0);

    EXPECT_EQ(zoneWithObstacleA.x, 1u);
    EXPECT_EQ(zoneWithObstacleA.y, 0u);

    EXPECT_EQ(zoneWithObstacleB.x, 1u) << "obstacle at pixel (65,45) should be counted in zone (6,4)";
    EXPECT_EQ(zoneWithObstacleB.y, 0u);

    EXPECT_EQ(untouchedZone.x, 0u);
    EXPECT_EQ(untouchedZone.y, 0u);
}

// // getSearchZonePtr() must expose the exact same data as
// // readSearchZoneInfo() for the zone the obstacle was placed in: reading
// // the raw pointer at that zone's flattened grid index should match the
// // accessor's result. Uses zone (3,1) on purpose so this also fails
// // against the current implementation, instead of only proving the
// // pointer and the accessor agree with each other on whatever (possibly
// // wrong) cell the kernel happened to write to.
TEST(TestSearchFrameSearchZone, GetSearchZonePtr_MatchesReadSearchZoneInfo)
{
    SearchFrame f(FRAME_W, FRAME_H, LOWER_BOUND, UPPER_BOUND, {ZONE_W, ZONE_H});
    setupObstacleClassCosts(f);

    // Zone (xg=3, zg=1) covers pixels x in [30,39], z in [10,19].
    std::vector<float> buf = buildBlankFrameBuffer();
    setObstaclePixel(buf, 35, 10); // border obstacle in zone (3, 1)
    f.copyFrom(buf.data());

    f.processSafeDistanceZone({ZONE_W, ZONE_H}, false);

    uint4 *ptr = f.getSearchZonePtr();
    ASSERT_NE(ptr, nullptr);

    int2 gridSize = f.getSearchZoneGridSize();
    const int gridW = gridSize.x;
    const int gridH = gridSize.y;
    const int zoneIndex = 1 * gridW + 3; // (xg=3, zg=1), per readSearchZoneInfo's own formula

    uint4 fromPtr = ptr[zoneIndex];
    uint4 fromAccessor = f.readSearchZoneInfo(3, 1);

    EXPECT_EQ(fromPtr.x, fromAccessor.x);
    EXPECT_EQ(fromPtr.y, fromAccessor.y);
    EXPECT_EQ(fromPtr.x, 1u);
    EXPECT_EQ(fromPtr.y, 1u);
}

// // Focused reproduction of the flat-index bug in
// // count_obstacle_in_search_zones() (src/cuda/search_zone_obstacle_count.cu).
// // The kernel computes the zone's flat storage index as
// //     posg = zg * WG + xg              (WG = zone width in pixels = 10)
// // but readSearchZoneInfo()/Frame::operator[] compute it as
// //     pos  = zg * GridWidth + xg       (GridWidth = 11 for this 100x100
// //                                        frame with 10x10 zones)
// // For zg=1, xg=2: posg = 1*10+2 = 12, which Frame<uint4> resolves back to
// // grid coordinates (12 % 11, 12 / 11) = (1, 1) -- NOT (2, 1). This test
// // places a single obstacle in zone (2,1) and shows:
// //   1) the intended zone (2,1) does NOT see it (fails today), and
// //   2) it is instead visible in zone (1,1), confirming the root cause.
// // Once count_obstacle_in_search_zones() is fixed to use the grid width
// // instead of the zone pixel width, assertion (1) should start passing
// // and assertion (2) should be removed/updated.
TEST(TestSearchFrameSearchZone, ReadSearchZoneInfo_NonFirstZoneRow_ExposesFlatIndexBug)
{
    SearchFrame f(FRAME_W, FRAME_H, LOWER_BOUND, UPPER_BOUND, {ZONE_W, ZONE_H});
    setupObstacleClassCosts(f);

    // Zone (xg=2, zg=1) covers pixels x in [20,29], z in [10,19].
    std::vector<float> buf = buildBlankFrameBuffer();
    setObstaclePixel(buf, 25, 15); // interior obstacle
    f.copyFrom(buf.data());

    f.processSafeDistanceZone({ZONE_W, ZONE_H}, false);

    uint4 intended = f.readSearchZoneInfo(2, 1);
    EXPECT_EQ(intended.x, 1u)
        << "BUG: obstacle at pixel (25,15) belongs to zone (2,1) but is not reported there. "
        << "count_obstacle_in_search_zones() likely wrote it to the wrong flat index "
        << "(see src/cuda/search_zone_obstacle_count.cu: posg = zg*WG+xg uses the zone's "
        << "pixel width instead of the zone grid width).";
    EXPECT_EQ(intended.y, 0u);

    int2 gridSize = f.getSearchZoneGridSize();
    const int gridW = gridSize.x;
    const int gridH = gridSize.y;

    const int buggyFlatIndex = 1 * ZONE_W + 2; // posg = zg*WG + xg, as computed by the kernel
    const int misplacedX = buggyFlatIndex % gridW;
    const int misplacedZ = buggyFlatIndex / gridW;

    uint4 misplaced = f.readSearchZoneInfo(misplacedX, misplacedZ);
    EXPECT_EQ(misplaced.x, 1u)
        << "diagnostic: obstacle intended for zone (2,1) actually landed in zone ("
        << misplacedX << "," << misplacedZ << ")";
}