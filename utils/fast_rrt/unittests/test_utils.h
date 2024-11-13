#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "../src/cuda_frame.h"

void dump_cuda_frame_to_file(CudaFrame *frame, const char *filename) {
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