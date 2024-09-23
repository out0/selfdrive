#include <gtest/gtest.h>
#include "../cuda_frame.h"
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>

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

using json = nlohmann::json;

void writeRGBToPNG(const unsigned char *data, int width, int height, const std::string &filename)
{
    // Create a Mat object to hold the image data
    cv::Mat image(height, width, CV_8UC3);

    // Copy the RGB data to the Mat object
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                image.at<cv::Vec3b>(y, x)[c] = data[(y * width + x) * 3 + c];
            }
        }
    }

    // Write the image to a PNG file
    cv::imwrite(filename, image);
}


TEST(SetGoalVectorizedTest, TestPositionShouldBeFeasibleForSomeDirection)
{
    // cv::Mat m = cv::imread("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/cudac/unittests/test_data/bev_1.png", cv::IMREAD_COLOR);
    cv::Mat m = cv::imread("test_data/bev_2.png", cv::IMREAD_COLOR);
    cv::Size s = m.size();
    // printf("BEV: %d x %d x %d type: %d\n", s.width, s.height, m.channels(), m.type());

    float *p = new float[256 * 256 * 3];
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++)
        {
            int pos = 3 * (256 * i + j);
            cv::Vec3b pixel = m.at<cv::Vec3b>(i, j);
            p[pos] = pixel[0];
            p[pos + 1] = pixel[1];
            p[pos + 2] = pixel[2];
        }

    CudaFrame f(
        p,
        s.width,
        s.height,
        MIN_DISTANCE_WIDTH_PX,
        MIN_DISTANCE_HEIGHT_PX,
        EGO_LOWER_BOUND_X,
        EGO_LOWER_BOUND_Z,
        EGO_UPPER_BOUND_X,
        EGO_UPPER_BOUND_Z);

    f.setGoalVectorized(31, 100);

    f.copyBack(p);



    u_char  *img = new u_char[256 * 256 * 3];

    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++)
        {
            int pos = 3 * (256 * i + j);
            img[pos] = (u_char)p[pos];
            img[pos + 1] = (u_char)p[pos+1];
            img[pos + 2] = (u_char)p[pos+2];
        }
     writeRGBToPNG(img, 256, 256, "result.png");
}


//     int x = 31;
//     int z = 100;

//     int pos = 3 * (256 * z + x);


//     ASSERT_EQ(p[pos], 1.0); // class should be 1.0
//     ASSERT_EQ(p[pos + 1], 0.0); // euclidian distance cost should be 0
//     ASSERT_TRUE(p[pos + 2] > 0.0); // should be feasible in some heading
// }
