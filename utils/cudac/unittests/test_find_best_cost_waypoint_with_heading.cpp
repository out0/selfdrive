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

TEST(WaypointFind, TestFindBestCostWaypointWithHeading)
{
    //cv::Mat image = cv::imread("test_data/bev_3.png", cv::IMREAD_UNCHANGED);
   

    CudaFrame * frame = TestFrame::readCudaFrameFromFile("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/cudac/unittests/test_data/bev_3.png",
    MIN_DISTANCE_WIDTH_PX, MIN_DISTANCE_HEIGHT_PX, EGO_LOWER_BOUND_X, EGO_LOWER_BOUND_Z, EGO_UPPER_BOUND_X, EGO_UPPER_BOUND_Z);


    float angle_rad = (3.141592654F * 14.56576424655659 / 180);

    float * waypoint = frame->bestWaypointPosForHeading(155, 16, angle_rad);

    ASSERT_EQ(waypoint[0], 122);
    ASSERT_EQ(waypoint[1], 8);
    ASSERT_FLOAT_EQ(waypoint[2], angle_rad);

    float *points = new float[4];
    points[0] = 122.0;
    points[1] = 8.0;
    points[2] = angle_rad;
    points[3] = 0;

    frame->checkFeasibleWaypoints(points, 1, false);

    ASSERT_FLOAT_EQ(points[3], 1.0);
}
