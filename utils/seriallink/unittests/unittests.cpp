#include <gtest/gtest.h>
#include <stdlib.h>


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    int p = RUN_ALL_TESTS();
    return p;
}