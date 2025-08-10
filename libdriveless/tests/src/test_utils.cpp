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


std::vector<Waypoint> testInterpolateHermiteCurve(int width, int height, Waypoint p1, Waypoint p2)
{
    std::vector<Waypoint> curve;

    // int numPoints = 2 * abs(max(int(p2.z() - p1.z()), int(p2.x() - p1.x()), 100));

    int numPoints = TO_INT(Waypoint::distanceBetween(p1, p2));

    // int numPoints = 100;
    curve.reserve(numPoints);

    // Distance between points (used to scale tangents)
    float dx = p2.x() - p1.x();
    float dz = p2.z() - p1.z();
    float d = sqrtf(dx * dx + dz * dz);

    float a1 = p1.heading().rad() - PI / 2;
    float a2 = p2.heading().rad() - PI / 2;

    // Tangent vectors
    float2 tan1 = {d * cosf(a1), d * sinf(a1)};
    float2 tan2 = {d * cosf(a2), d * sinf(a2)};

    int last_x = -1;
    int last_z = -1;

    for (int i = 0; i < numPoints; ++i)
    {
        float t = static_cast<float>(i) / (numPoints - 1);

        float t2 = t * t;
        float t3 = t2 * t;

        // Hermite basis functions
        float h00 = 2 * t3 - 3 * t2 + 1;
        float h10 = t3 - 2 * t2 + t;
        float h01 = -2 * t3 + 3 * t2;
        float h11 = t3 - t2;

        float x = h00 * p1.x() + h10 * tan1.x + h01 * p2.x() + h11 * tan2.x;
        float z = h00 * p1.z() + h10 * tan1.y + h01 * p2.z() + h11 * tan2.y;

        if (x < 0 || x >= width)
            continue;
        if (z < 0 || z >= height)
            continue;

        int cx = static_cast<int>(round(x));
        int cz = static_cast<int>(round(z));

        if (cx == last_x && cz == last_z)
            continue;
        if (cx < 0 || cx >= width)
            continue;
        if (cz < 0 || cz >= height)
            continue;

        float t00 = 6 * t2 - 6 * t;
        float t10 = 3 * t2 - 4 * t + 1;
        float t01 = -6 * t2 + 6 * t;
        float t11 = 3 * t2 - 2 * t;

        float ddx = t00 * p1.x() + t10 * tan1.x + t01 * p2.x() + t11 * tan2.x;
        float ddz = t00 * p1.z() + t10 * tan1.y + t01 * p2.z() + t11 * tan2.y;

        float heading = atan2f(ddz, ddx) + HALF_PI;

        // Interpolated point
        curve.push_back({cx, cz, angle::rad(heading)});
        last_x = cx;
        last_z = cz;
    }

    return curve;
}