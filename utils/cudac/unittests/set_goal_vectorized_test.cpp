#include "../include/cuda_frame.h"
#include "test_frame.h"

#define MIN_DISTANCE_WIDTH_PX 22
#define MIN_DISTANCE_HEIGHT_PX 40
#define EGO_LOWER_BOUND_X 119
#define EGO_LOWER_BOUND_Z 148
#define EGO_UPPER_BOUND_X 137
#define EGO_UPPER_BOUND_Z 108

typedef struct waypoint
{
    int x;
    int y;
} waypoint;




TEST(SetGoalVectorizedTest, FreeZoneInside)
{
    TestFrame tst_frame(256, 256);

    CudaFrame * frame = tst_frame.getCudaFrame(MIN_DISTANCE_WIDTH_PX, MIN_DISTANCE_HEIGHT_PX, EGO_LOWER_BOUND_X, EGO_LOWER_BOUND_Z, EGO_UPPER_BOUND_X, EGO_UPPER_BOUND_Z);
    frame->setGoalVectorized(100, 50);
    frame->copyBack(tst_frame.getImgPtr());
    tst_frame.toFile("test.png");
}


//     int x = 31;
//     int z = 100;

//     int pos = 3 * (256 * z + x);


//     ASSERT_EQ(p[pos], 1.0); // class should be 1.0
//     ASSERT_EQ(p[pos + 1], 0.0); // euclidian distance cost should be 0
//     ASSERT_TRUE(p[pos + 2] > 0.0); // should be feasible in some heading
// }
