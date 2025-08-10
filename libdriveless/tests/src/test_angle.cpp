#include <gtest/gtest.h>
#include "../../include/angle.h"
#include "test_utils.h"
#include <cmath>

TEST(Angle, TestBooleanOperators)
{
    angle a = angle::deg(12.2);
    angle v = angle::deg(12.2);
    
    ASSERT_TRUE(a == v);
    ASSERT_TRUE(a <= v);
    ASSERT_TRUE(a >= v);
    v.setDeg(12.4);
    ASSERT_FALSE(a == v);
    ASSERT_TRUE(a != v);
    ASSERT_TRUE(a < v);
    ASSERT_TRUE(a <= v);
    ASSERT_TRUE(v > a);
    ASSERT_TRUE(v >= a);
}

TEST(Angle, TestCompoundOperators)
{
    angle a = angle::deg(12.2);
    angle v = angle::deg(12.3);
    
    angle c = a + v;
    ASSERT_DEQ(24.5, c.deg());

    c = a - v;
    ASSERT_DEQ(-0.1, c.deg());

    ASSERT_DEQ(6.1, (a / 2).deg());
    ASSERT_TRUE(angle::deg(6.1) == (a / 2));
}


TEST(Angle, TestConversion)
{
    double a = 0, a_rad = 0;

    while (a < 360) {
        auto b = angle::deg(a);
        a_rad = (a * PI)/180;
        ASSERT_DEQ(a_rad, b.rad());
        ASSERT_DEQ(a, b.deg());
        a += 0.0001;
    }    
}

TEST(Angle, TestRadianConversion)
{
    ASSERT_FLOAT_EQ(angle::deg(0).rad(), 0);
    ASSERT_FLOAT_EQ(angle::deg((double)45/2).rad(), PI/8);
    ASSERT_FLOAT_EQ(angle::deg(45).rad(), PI/4);
    ASSERT_FLOAT_EQ(angle::deg(67.5).rad(), 3*PI/8);
    ASSERT_FLOAT_EQ(angle::deg(90).rad(), PI/2);
}