#pragma once

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "../src/cuda_frame.h"
#include "../include/physical_parameters.h"
#include "../src/class_def.h"
#include "../src/kinematic_model.h"

void dump_cuda_frame_to_file(CudaFrame *frame, const char *filename)
{
    cv::Mat cimg = cv::Mat(frame->getHeight(), frame->getWidth(), CV_8UC3);
    float3 *ptr = frame->getFramePtr();

    for (int i = 0; i < frame->getHeight(); i++)
        for (int j = 0; j < frame->getWidth(); j++)
        {
            cv::Vec3b &pixel = cimg.at<cv::Vec3b>(i, j);
            int idx = i * 256 + j;

            // Set the channels: Blue, Green, Red
            pixel[0] = static_cast<int>(round(ptr[idx].x)); // Blue
            pixel[1] = static_cast<int>(round(ptr[idx].y)); // Green
            pixel[2] = static_cast<int>(round(ptr[idx].z)); // Red
        }

    cv::imwrite(filename, cimg);
}

float *create_matrix(int rows, int cols, int channels, float fill_val)
{
    int count = rows * cols * channels;
    float *mat = new float[count];
    for (int i = 0; i < count; i++)
        mat[i] = fill_val;
    return mat;
}

CudaFrame *create_default_cuda_frame(float fill_val)
{
    float *mat = create_matrix(OG_WIDTH, OG_HEIGHT, 3, fill_val);

    return new CudaFrame(mat, OG_WIDTH, OG_HEIGHT, MIN_DIST_X, MIN_DIST_Z, LOWER_BOUND_X, LOWER_BOUND_Z, UPPER_BOUND_X, UPPER_BOUND_Z);
}