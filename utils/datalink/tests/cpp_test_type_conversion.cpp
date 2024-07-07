#include <gtest/gtest.h>
#include <stdlib.h>
#include "../datalink.h"

TEST(TypeConversionTest, TestEncodeUint82)
{
    uint8 p;
    p.val = 23;

    uint8 q;
    q.bval = p.val;

    ASSERT_EQ(p.val, q.val);    
    printf("%d\n", p.bval);

    p.val = 230;
    q.bval = p.val;
    printf("%d\n", p.bval);
    ASSERT_EQ(p.val, q.val);
}
