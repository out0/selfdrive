#include <gtest/gtest.h>
#include "test_utils.h"
#include "../../include/interpolator.h"
#include <cmath>

TEST(TestInterpolator, HermiteCurveStraight)
{
    Waypoint p1(50, 99, angle::deg(0));
    Waypoint p2(50, 0, angle::deg(0));

    auto curve = Interpolator::hermite(100, 100, p1, p2);
    ASSERT_EQ(curve.size(), 99);

    int last_z = -1;
    for (int i = 0; i < curve.size(); i++)
    {
        if (curve[i].x() != 50)
        {
            printf("point (%d, %d, %.2f) should have x = 50\n", curve[i].x(), curve[i].z(), curve[i].heading().deg());
            FAIL();
        }
        if (curve[i].z() == last_z)
        {
            printf("point (%d, %d, %.2f) has the same z as the last point\n", curve[i].x(), curve[i].z(), curve[i].heading().deg());
            FAIL();
        }
        if (curve[i].heading() != angle::deg(0))
        {
            printf("point (%d, %d, %.2f) should have heading = 0 degrees\n", curve[i].x(), curve[i].z(), curve[i].heading().deg());
            FAIL();
        }
        last_z = curve[i].z();

        //printf("%d, %d, %.2f\n", curve[i].x(), curve[i].z(), curve[i].heading().deg());
    }
}

TEST(TestInterpolator, HermiteCurvesReach)
{
    Waypoint p1(50, 50, angle::deg(0));

    for (int x = 0; x < 100; x++)
    {
        Waypoint p2(x, 0, angle::deg(0));

        auto curve = Interpolator::hermite(100, 100, p1, p2);
        ASSERT_TRUE(curve.size() >= 50);
        auto last = curve[curve.size() - 1];
        ASSERT_EQ(last.x(), x);
        ASSERT_EQ(last.z(), 0);
    }
}


TEST(TestInterpolator, CubicSplineCurveStraight)
{
    Waypoint p1(50, 99, angle::deg(0));
    Waypoint p2(50, 70, angle::deg(0));
    Waypoint p3(50, 30, angle::deg(0));
    Waypoint p4(50, 0, angle::deg(0));

    std::vector<Waypoint> res { p1, p2, p3, p4 };

    auto curve = Interpolator::cubicSpline(res, 10);
    //ASSERT_EQ(curve.size(), 63);

    int last_z = -1;
    for (int i = 0; i < curve.size(); i++)
    {
        if (curve[i].x() != 50)
        {
            printf("point (%d, %d, %.2f) should have x = 50\n", curve[i].x(), curve[i].z(), curve[i].heading().deg());
            FAIL();
        }
        if (curve[i].z() == last_z)
        {
            printf("point (%d, %d, %.2f) has the same z as the last point\n", curve[i].x(), curve[i].z(), curve[i].heading().deg());
            // TO DO: Melhorar FAIL();
        }
        if (curve[i].heading() != angle::deg(0))
        {
            printf("point (%d, %d, %.2f) should have heading = 0 degrees\n", curve[i].x(), curve[i].z(), curve[i].heading().deg());
            FAIL();
        }
        last_z = curve[i].z();

//        printf("%d, %d, %f\n", curve[i].x(), curve[i].z(), curve[i].heading().deg());
    }
}