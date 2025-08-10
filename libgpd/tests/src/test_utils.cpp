#include <cmath>
#include <stdio.h>
#include <driveless/search_frame.h>
#include <opencv2/opencv.hpp>
bool _ASSERT_DEQ(double a, double b, int tolerance = 4) {
    double p = pow(10, -tolerance);
    
    if (abs(a - b) > p) {
        printf("ASSERT_DEQ failed: %f != %f, tolerance: %f\n", a, b, p);
        return false;
    }

    return true;
}

void dump_search_frame_debug(SearchFrame &f1)
{
    cv::Mat img(100, 100, CV_8UC3);

    for (int y = 0; y < 100; ++y)
        for (int x = 0; x < 100; ++x)
        {
            auto v = f1[{x, y}];
            cv::Vec3b color;
            if ((int)v.x == 1)
            {
                color[0] = 0;
                color[1] = 0;
                color[2] = 0;
            }
            else
            {
                color[0] = 255;
                color[1] = 255;
                color[2] = 255;
            }
            if (((int)v.z & 0x100) > 0)
            {
                color[0] = 0;
                color[1] = 255;
                color[2] = 0;
            }
            if (((int)v.z & 0x200) > 0)
            {
                if (!((int)v.z & 0x100) > 0)
                {
                    color[0] = 192;
                    color[1] = 192;
                    color[2] = 192;
                }
                else
                {
                    color[0] = 0;
                    color[1] = 0;
                    color[2] = 255;
                }
            }
            img.at<cv::Vec3b>(y, x) = color;
        }

    cv::imwrite("debug.png", img);
}
