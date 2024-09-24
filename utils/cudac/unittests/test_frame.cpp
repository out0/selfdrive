#include <gtest/gtest.h>
#include "../cuda_frame.h"
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "test_frame.h"

TestFrame::TestFrame(int width, int height)
{
    int size = width * height * 3;
    this->data = new float[size];
    this->width = width;
    this->height = height;

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                data[(y * width + x) * 3 + c] = 255;
}
TestFrame::~TestFrame()
{
    delete[] data;
}

int TestFrame::getVectorPos(int x, int z)
{
    return 3 * (width * z + x);
}

void TestFrame::addObstacle(int x1, int z1, int x2, int z2)
{
    for (int z = z1; z <= z2; z++)
        for (int x = x1; x <= x2; x++)
        {
            int pos = getVectorPos(x, z);

            data[pos] = 0;
            data[pos + 1] = 0;
            data[pos + 2] = 0;
        }
}

void TestFrame::addBluePoint(int x, int z)
{
    addPoint(x, z, 255, 0, 0);
}

void TestFrame::addRedPoint(int x, int z)
{
    addPoint(x, z, 0, 0, 255);
}

void TestFrame::addGreenPoint(int x, int z)
{
    addPoint(x, z, 0, 255, 0);
}

float *TestFrame::getImgPtr()
{
    return data;
}

float TestFrame::getDistanceCost(int x, int z)
{
    int pos = getVectorPos(x, z);
    return data[pos + 1];
}
int TestFrame::getAllowedHeadings(int x, int z)
{
    int pos = getVectorPos(x, z);
    return (int)data[pos + 2];
}

void TestFrame::addPoint(int x, int z, int r, int g, int b)
{
    for (int j = z - 1; j <= z + 1; j++)
    {
        if (j < 0 || j >= height)
            continue;

        int pos = getVectorPos(x, j);
        data[pos] = r;
        data[pos + 1] = g;
        data[pos + 2] = b;
    }
    for (int i = x - 1; i <= x + 1; i++)
    {
        if (i < 0 || i >= width)
            continue;

        int pos = getVectorPos(i, z);
        data[pos] = r;
        data[pos + 1] = g;
        data[pos + 2] = b;
    }
}

void addVector(int x, int z, float heading, int size)
{
}

CudaFrame *TestFrame::getFrame(int min_dist_x, int min_dist_z, int lower_bound_x, int lower_bound_z, int upper_bound_x, int upper_bound_z)
{
    for (int z = 0; z < height; z++)
    {
        for (int x = 0; x < width; x++)
        {
            int pos = getVectorPos(x, z);
            if (data[pos] != 0)
            {
                data[pos] = 1;
                data[pos + 1] = 0;
                data[pos + 2] = 0;
            }
        }
    }

    return new CudaFrame(
        this->data, this->width, this->height, min_dist_x, min_dist_z, lower_bound_x, lower_bound_z, upper_bound_x, upper_bound_z);
}

void TestFrame::toFile(const std::string &filename)
{
    // Create a Mat object to hold the image data
    cv::Mat image(height, width, CV_8UC3);

    // Copy the RGB data to the Mat object
    for (int z = 0; z < height; z++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                int val = data[(z * width + x) * 3 + c];
                image.at<cv::Vec3b>(z, x)[c] = val;
            }
        }
    }

    // Write the image to a PNG file
    cv::imwrite(filename, image);
}
