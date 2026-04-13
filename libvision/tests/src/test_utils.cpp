#include <cmath>
#include <stdio.h>
#include <vector>
#include <driveless/waypoint.h>
#include <driveless/angle.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "test_utils.h"

bool _ASSERT_DEQ(double a, double b, int tolerance) {
    double p = pow(10, -tolerance);
    
    if (abs(a - b) > p) {
        printf("ASSERT_DEQ failed: %f != %f, tolerance: %f\n", a, b, p);
        return false;
    }

    return true;
}

void exportSearchFrameToFile(SearchFrame &f, const char *file) {

    int size = f.width() * f.height() * 3;

    uchar *outp = new uchar[size];
    f.exportToColorFrame(outp);

    cv::Mat rgb_image(f.width(), f.height(), CV_8UC3, outp);

    // Convert RGB to BGR
    cv::Mat bgr_image;
    cv::cvtColor(rgb_image, bgr_image, cv::COLOR_RGB2BGR);

    // Save to file
    if (cv::imwrite(file, bgr_image))
    {
        std::cout << "Image saved successfully.\n";
    }
    else
    {
        std::cerr << "Failed to save image.\n";
    }

    // Clean up (if f was dynamically allocated)
    delete[] outp;
}

