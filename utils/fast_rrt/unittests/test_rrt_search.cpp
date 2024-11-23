#include "../src/cuda_graph.h"
#include "../src/cuda_frame.h"
#include "../include/physical_parameters.h"
#include "../include/fast_rrt.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <vector>

TEST(FastRRTTst, TestRRTSearch)
{
    cv::Mat img = cv::imread("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/fast_rrt/unittests/bev_1.png", cv::IMREAD_COLOR);
    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels();
    float *ptr = new float[rows * cols * channels];

    for (int z = 0; z < rows; z++)
    {
        for (int x = 0; x < cols; x++)
        {
            int pos = 3 * (z * cols + x);
            cv::Vec3b pixel = img.at<cv::Vec3b>(z, x);
            ptr[pos] = pixel[0];
            ptr[pos + 1] = pixel[1];
            ptr[pos + 2] = pixel[2];

            // if (ptr[pos] > 0)
            //     printf ("x = %f\n", ptr[pos]);
        }
    }

    CudaFrame cuda_frame(ptr,
                         OG_WIDTH,
                         OG_HEIGHT,
                         MIN_DIST_X,
                         MIN_DIST_Z,
                         LOWER_BOUND_X,
                         LOWER_BOUND_Z,
                         UPPER_BOUND_X,
                         UPPER_BOUND_Z);

    FastRRT rrt(OG_WIDTH,
                OG_HEIGHT,
                OG_REAL_WIDTH,
                OG_REAL_HEIGHT,
                MIN_DIST_X,
                MIN_DIST_Z,
                LOWER_BOUND_X,
                LOWER_BOUND_Z,
                UPPER_BOUND_X,
                UPPER_BOUND_Z,
                MAX_STEERING_ANGLE,
                -1);

    double3 start = {128, 128, 0}, end = {115, 0, 0.6931};
    rrt.setPlanData(cuda_frame.getFramePtr(), start, end, 1.0);
    rrt.search();

    int p = 2;
}
