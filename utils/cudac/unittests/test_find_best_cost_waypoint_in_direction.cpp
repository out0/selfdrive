#include "../include/cuda_frame.h"
#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>
#include "test_frame.h"

#define MIN_DISTANCE_WIDTH_PX 22
#define MIN_DISTANCE_HEIGHT_PX 40
#define EGO_LOWER_BOUND_X 119
#define EGO_LOWER_BOUND_Z 148
#define EGO_UPPER_BOUND_X 137
#define EGO_UPPER_BOUND_Z 108

class Waypoint {
public:
    int x;
    int z;
    float heading_deg;

    Waypoint(int x, int z, float heading_deg) : x(x), z(z), heading_deg(heading_deg) {}
};

TEST(WaypointFindInDirection, TestFindBestCostWaypointInDirection)
{
    return;
    //cv::Mat image = cv::imread("test_data/bev_3.png", cv::IMREAD_UNCHANGED);
    CudaFrame * frame = TestFrame::readCudaFrameFromFile("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/cudac/unittests/test_data/bev_4.png",
    MIN_DISTANCE_WIDTH_PX, MIN_DISTANCE_HEIGHT_PX, EGO_LOWER_BOUND_X, EGO_LOWER_BOUND_Z, EGO_UPPER_BOUND_X, EGO_UPPER_BOUND_Z);

    float * waypoint = frame->bestWaypointPos(130, -20);

    Waypoint point((int)waypoint[0], (int)waypoint[1], waypoint[2]);
    delete []waypoint;


    waypoint = frame->bestWaypointInDirection(128, 107, 130, -20);


    ASSERT_EQ(waypoint[0], point.x);
    ASSERT_EQ(waypoint[1], point.z);
    ASSERT_FLOAT_EQ(waypoint[3], point.heading_deg);
}

TEST(WaypointFindInDirection, TestFindBestCostWaypointInDirection2)
{
    //cv::Mat image = cv::imread("test_data/bev_3.png", cv::IMREAD_UNCHANGED);
    CudaFrame * frame = TestFrame::readCudaFrameFromFile("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/cudac/unittests/test_data/bev_5.png",
    MIN_DISTANCE_WIDTH_PX, MIN_DISTANCE_HEIGHT_PX, EGO_LOWER_BOUND_X, EGO_LOWER_BOUND_Z, EGO_UPPER_BOUND_X, EGO_UPPER_BOUND_Z);

    float * waypoint = frame->bestWaypointPos(128, -41);

    Waypoint point((int)waypoint[0], (int)waypoint[1], waypoint[2]);
    delete []waypoint;


    waypoint = frame->bestWaypointInDirection(128, 107, 128, -41);

    ASSERT_TRUE(waypoint[0] >= 0.9 * point.x && waypoint[0] <= 1.1 * point.x);
    ASSERT_TRUE(waypoint[1] >= 0 && waypoint[1] < 10);
    ASSERT_FLOAT_EQ(waypoint[3], point.heading_deg);
}


TEST(WaypointFindInDirection, TestFindBestCostWaypointInDirection3)
{
    //cv::Mat image = cv::imread("test_data/bev_3.png", cv::IMREAD_UNCHANGED);
    CudaFrame * frame = TestFrame::readCudaFrameFromFile("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/cudac/unittests/test_data/bev_6.png",
    MIN_DISTANCE_WIDTH_PX, MIN_DISTANCE_HEIGHT_PX, EGO_LOWER_BOUND_X, EGO_LOWER_BOUND_Z, EGO_UPPER_BOUND_X, EGO_UPPER_BOUND_Z);

    float * waypoint = frame->bestWaypointPos(167, -56);

    Waypoint point((int)waypoint[0], (int)waypoint[1], waypoint[2]);
    delete []waypoint;


    waypoint = frame->bestWaypointInDirection(128, 107, 167, -56);

    ASSERT_TRUE(waypoint[0] >= 0.9 * point.x && waypoint[0] <= 1.1 * point.x);
    ASSERT_TRUE(waypoint[1] >= 0 && waypoint[1] < 10);
    ASSERT_FLOAT_EQ(waypoint[3], point.heading_deg);
}
