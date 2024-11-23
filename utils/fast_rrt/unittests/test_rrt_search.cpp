#include "../src/cuda_graph.h"
#include "../src/cuda_frame.h"
#include "../src/class_def.h"
#include "../src/kinematic_model.h"
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



void show_point(cv::Mat &mat, double x, double z, int r, int g, int b)
{
    if (x < 0 || x >= mat.cols)
        return;
    if (z < 0 || z >= mat.rows)
        return;
    cv::Vec3b &pixel = mat.at<cv::Vec3b>(static_cast<int>(z), static_cast<int>(x));
    pixel[0] = r;
    pixel[1] = g;
    pixel[2] = b;
}

void write_planned_path_to_file(CudaFrame &frame, std::vector<double3> path, const char *filename, double velocity_m_s)
{
    cv::Mat image(frame.getHeight(), frame.getWidth(), CV_8UC3);

    auto og = frame.getFramePtr();

    for (int z = 0; z < frame.getHeight(); z++)
        for (int x = 0; x < frame.getWidth(); x++)
        {
            int c = og[z * frame.getWidth() + x].x;
            cv::Vec3b &pixel = image.at<cv::Vec3b>(z, x);
            pixel[0] = segmentationClassColors[c][0];
            pixel[1] = segmentationClassColors[c][1];
            pixel[2] = segmentationClassColors[c][2];
        }

    double rw = OG_WIDTH / OG_REAL_WIDTH;
    double rh = OG_HEIGHT / OG_REAL_HEIGHT;

    double3 center = {
        static_cast<int>(round(OG_WIDTH / 2)),
        static_cast<int>(round(OG_REAL_HEIGHT / 2)),
        0.0};

    double lr = 0.5 * (LOWER_BOUND_Z - UPPER_BOUND_Z) / (OG_HEIGHT / OG_REAL_HEIGHT);

    double3 end = {-1, -1, -1};

    for (double3 start : path)
    {
        if (end.x >= 0 && end.y >= 0)
        {
            // std::vector<double3> curve = CurveGenerator::buildCurveWaypoints(center, rw, rh, lr, MAX_STEERING_ANGLE, start, end, velocity_m_s);

            // for (double3 point : curve)
            // {
            //     show_point(image, point.x, point.y, 255, 255, 255);
            // }

            show_point(image, start.x, start.y, 0, 0, 255);
            show_point(image, end.x, end.y, 0, 0, 255);
        }
        end.x = start.x;
        end.y = start.y;
        end.z = start.z;
    }

    // Save the image to verify the change
    cv::imwrite(filename, image);
}


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

    int i = 1;
    while (i <= 200)
    {
        printf("Search #%d\n", i);
        double3 start = {128, 128, 0}, end = {115, 0, 0.6931};
        rrt.setPlanData(cuda_frame.getFramePtr(), start, end, 1.0);
        rrt.search();
        std::vector<double3> &path = rrt.getPath();
        write_planned_path_to_file(cuda_frame, path, "plan_output.png", 1.0);
        i++;
    }
    
}
