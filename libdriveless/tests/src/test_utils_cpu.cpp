#include <cmath>
#include <stdio.h>
#include <vector>
#include <driveless/waypoint.h>
#include <driveless/angle.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "test_utils_cpu.h"

extern bool _ASSERT_DEQ(double a, double b, int tolerance);
extern std::vector<Waypoint> testInterpolateHermiteCurve(int width, int height, Waypoint p1, Waypoint p2);

void exportSearchFrameCPUToFile(SearchFrameCPU &f, const char *file, std::vector<Waypoint> path) {

    int size = f.width() * f.height() * 3;

    uchar *outp = new uchar[size];
    f.exportToColorFrame(outp);

    cv::Mat rgb_image(f.width(), f.height(), CV_8UC3, outp);

    // Convert RGB to BGR
    cv::Mat bgr_image;
    cv::cvtColor(rgb_image, bgr_image, cv::COLOR_RGB2BGR);

    for (auto p : path) {
        bgr_image.at<cv::Vec3b>(p.z(), p.x()) = cv::Vec3b(0, 0, 255);
    }

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


