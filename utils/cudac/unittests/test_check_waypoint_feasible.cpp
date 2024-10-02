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

TEST(CheckWaypointFeasible, CheckPointFeasible)
{
    // cv::Mat image = cv::imread("test_data/bev_3.png", cv::IMREAD_UNCHANGED);
    CudaFrame * frame = TestFrame::readCudaFrameFromFile("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/cudac/unittests/test_data/bev_3.png",
    MIN_DISTANCE_WIDTH_PX, MIN_DISTANCE_HEIGHT_PX, EGO_LOWER_BOUND_X, EGO_LOWER_BOUND_Z, EGO_UPPER_BOUND_X, EGO_UPPER_BOUND_Z);
    
    float *points = new float[4];
    points[0] = 155.0;
    points[1] = 16.0;
    points[2] = 9.46232220802561;
    points[3] = 1;

    frame->checkFeasibleWaypoints(points, 1, false);

    ASSERT_EQ(points[3], 0);
}
