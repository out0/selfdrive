#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "test_utils.h"
#include "../../include/fastrrt.h"
#include "../../../cudac/include/cuda_frame.h"
#include "tst_class_def.h"
//#include <driveless/cubic_interpolator.h>
#include "../../include/waypoint.h"

#define PHYS_SIZE 34.641016151377535

std::pair<cv::Mat, float *> readImg(const char *file)
{
    cv::Mat img = cv::imread(file, cv::IMREAD_COLOR);
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

    return std::pair<cv::Mat, float *>(img, ptr);
}

#define OG_REAL_WIDTH 34.641016151377535
#define OG_REAL_HEIGHT 34.641016151377535
#define MAX_STEERING_ANGLE 40
#define VEHICLE_LENGTH_M 5.412658774


void exportPathTo(cudaPtr f, int width, int height, std::vector<Waypoint> &path, const char *file)
{
    uchar *dest = new uchar[3 * width * height];

    CudaFrame frame(f, width, height, 0, 0, -1, -1, -1, -1);
    frame.convertToColorFrame(dest);

    for (auto p : path)
    {
        int pos = 3 * (width * p.z() + p.x());
        dest[pos] = 0;
        dest[pos + 1] = 0;
        dest[pos + 2] = 255;
    }

    cv::Mat cimg = cv::Mat(height, width, CV_8UC3, cv::Scalar(0));

    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
        {
            long pos = 3 * (h * width + w);
            cv::Vec3b &pixel = cimg.at<cv::Vec3b>(h, w);
            pixel[0] = dest[pos];
            pixel[1] = dest[pos + 1];
            pixel[2] = dest[pos + 2];
        }

    cv::imwrite(file, cimg);
    delete []dest;
}

#define TIMEOUT 20000

TEST(TestRRT, TestSearch)
{
    std::pair<cv::Mat, float *> res = readImg("/home/cristiano/Documents/Projects/Mestrado/code/driveless-new/libfastrrt/tests/bev_1.png");
    cv::Mat img = res.first;
    float *ptr = res.second;

    CudaFrame frame(ptr, img.cols, img.rows, 22, 40, 119, 148, 137, 108);
    //SearchFrame frame(img.cols, img.rows, {22, 40}, {119, 148}, {137, 108});
    //frame.copyFrom(ptr);

    //frame.setClassCosts(classCosts);
    //frame.setClassColors(classColors);

    float maxPathSize = 20.0;
    float distToGoal = 20.0;

    FastRRT rrt(
        img.cols, img.rows,
        OG_REAL_WIDTH, OG_REAL_HEIGHT,
        angle::deg(MAX_STEERING_ANGLE),
        VEHICLE_LENGTH_M,
        TIMEOUT,
        {22, 40},
        {119, 148},
        {137, 108},
        maxPathSize,
        distToGoal);

    ASSERT_FALSE(rrt.isPlanning());

    std::vector<Waypoint> path = rrt.getPlannedPath();

    ASSERT_EQ(path.size(), 0);

    Waypoint goal(128, 0, angle::rad(0));
    rrt.setPlanData(frame.getFramePtr(), &goal, 1);

    rrt.run();

    ASSERT_TRUE(rrt.goalReached());

    path = rrt.getPlannedPath();
    //auto interpol_path = CubicInterpolator::cubicSplineInterpolation(path, 50);
    exportPathTo(frame.getFramePtr(), img.cols, img.rows, path, "output1.png");

    // rrt.optimize();
    // path = rrt.getPlannedPath();
    // //interpol_path = CubicInterpolator::cubicSplineInterpolation(path, 50);
    // exportPathTo(&frame, path, "output2.png");

    ASSERT_TRUE(rrt.goalReached());

    delete[] ptr;
}
