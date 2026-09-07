#include <gtest/gtest.h>
#include "../../include/search_frame.h"
#include "../../include/interpolator.h"
#include "test_utils.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <tuple>
#include <vector>

std::tuple<int, int> minMaxCurve(std::vector<Waypoint> curve, int z)
{
    int x_max = -1;
    int x_min = 999999;

    for (int i = 0; i < curve.size(); i++)
    {
        Waypoint p = curve[i];
        if (p.z() == z)
        {
            // printf("processing %d,%d at %d\n", p.x(), p.z(), i);
            if (p.x() < x_min)
                x_min = p.x();

            if (p.x() > x_max)
                x_max = p.x();
        }
    }

    return {x_min, x_max};
}

TEST(TestSearchFrameExclusionZone, TestKinematicExclusionZoneReachableArea)
{
    SearchFrame f1(800, 800, {5, 5}, {15, 15});

    f1.setClassCosts({{1.0},
                      {-1.0}});

    f1.setPhysicalDimensionInMeters(100, 100);
    f1.setVehicleParams(5.34, angle::deg(40));

    const int SIZE = 3 * f1.width() * f1.height();
    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, false);

    // Every position must be traversable
    for (int z = 0; z < f1.height(); z++)
        for (int x = 0; x < f1.width(); x++)
            if (!f1.isTraversable(x, z))
            {
                printf("it should be traversable at (%d, %d)\n", x, z);
                FAIL();
            }

    Waypoint origin(400, 799, angle::deg(0));
    Waypoint goal(400, 0, angle::deg(0));

    // Disable kinematic invalid positions
    f1.processKinematicExclusionAreas(origin, goal);

    std::vector<Waypoint> curve1 = Interpolator::kinematicCurve(f1.width(), f1.height(), origin, angle::deg(-40), 5.34 * 8, -1);
    std::vector<Waypoint> curve2 = Interpolator::kinematicCurve(f1.width(), f1.height(), origin, angle::deg(40), 5.34 * 8, -1);

    goal.set_heading(angle::deg(180).rad());
    std::vector<Waypoint> curve3 = Interpolator::kinematicCurve(f1.width(), f1.height(), goal, angle::deg(-40), 5.34 * 8, -1);
    std::vector<Waypoint> curve4 = Interpolator::kinematicCurve(f1.width(), f1.height(), goal, angle::deg(40), 5.34 * 8, -1);

    std::vector<std::vector<Waypoint>> lst = {curve1, curve2, curve3, curve4};

    std::vector<uchar> dest(static_cast<size_t>(f1.width()) * f1.height() * 3);
    cv::Mat cimg(f1.height(), f1.width(), CV_8UC3, dest.data());

    // for (std::vector<Waypoint> curve : lst)
    // {
    //     for (Waypoint p : curve)
    //     {
    //         cv::Vec3b &pixel = cimg.at<cv::Vec3b>(p.z(), p.x());
    //         pixel[0] = 0;
    //         pixel[1] = 255;
    //         pixel[2] = 0;
    //     }
    // }

    std::vector<std::pair<std::vector<Waypoint>, std::vector<Waypoint>>> cones = {
        {curve1, curve2},
        {curve3, curve4}};

    for (int z = 0; z < f1.height(); z++)
    {
        for (auto &cone : cones)
        {
            auto [min1, max1] = minMaxCurve(cone.first, z);
            auto [min2, max2] = minMaxCurve(cone.second, z);

            bool curve1HasPoint = (max1 != -1);
            bool curve2HasPoint = (max2 != -1);

            if (!curve1HasPoint && !curve2HasPoint)
                continue;

            for (int x = 0; x < f1.width(); x++)
            {
                bool insideCurve1 = curve1HasPoint && x >= min1 && x <= max1;
                bool insideCurve2 = curve2HasPoint && x >= min2 && x <= max2;

                // Opposite of "on either curve" — NOT merged into one span
                if (!insideCurve1 && !insideCurve2)
                {
                    cv::Vec3b &pixel = cimg.at<cv::Vec3b>(z, x);
                    pixel[0] = 128;
                    pixel[1] = 128;
                    pixel[2] = 128;

                    if (!f1.isTraversable(x, z))
                    {
                        printf("should be traversable at %d,%d\n", x, z);
                        cv::Vec3b &pixel = cimg.at<cv::Vec3b>(z, x);
                        pixel[0] = 0;
                        pixel[1] = 0;
                        pixel[2] = 255;
                    }
                }
            }
        }
    }

    /*
    for (int z = 0; z < f1.height(); z++)
    {
        for (auto curve : lst)
        {
            auto [min, max] = minMaxCurve(curve, z);
            if (max == -1 || min > f1.width())
                continue;

            for (int x = 0; x < f1.width(); x++)
            {
                // if (!f1.isTraversable(x, z) && (x < min || x > max))
                if (x >= min && x <= max)
                {
                    if (f1.isTraversable(x, z))
                    {
                        printf("should NOT be traversable at %d,%d\n", x, z);
                        cv::Vec3b &pixel = cimg.at<cv::Vec3b>(z, x);
                        pixel[0] = 128;
                        pixel[1] = 0;
                        pixel[2] = 128;
                    }
                }
            }
        }
    }*/

    // for (std::vector<Waypoint> curve : lst)
    // {
    //     for (Waypoint p : curve)
    //     {
    //         cv::Vec3b &pixel = cimg.at<cv::Vec3b>(p.z(), p.x());
    //         pixel[0] = 0;
    //         pixel[1] = 0;
    //         pixel[2] = 0;
    //     }
    // }

    cv::imwrite("outp.png", cimg);

    int l = 1;
}
TEST(TestSearchFrameExclusionZone, TestKinematicExclusionZoneUnreachableArea)
{
    return;
    SearchFrame f1(800, 800, {5, 5}, {15, 15});

    f1.setClassCosts({{1.0},
                      {-1.0}});

    f1.setPhysicalDimensionInMeters(100, 100);
    f1.setVehicleParams(5.34, angle::deg(40));

    const int SIZE = 3 * f1.width() * f1.height();
    float *ptr = new float[SIZE];
    memset(ptr, 0x0, sizeof(float) * SIZE);
    f1.copyFrom(ptr);

    f1.processSafeDistanceZone({10, 10}, false);

    // Every position must be traversable
    for (int z = 0; z < f1.height(); z++)
        for (int x = 0; x < f1.width(); x++)
            if (!f1.isTraversable(x, z))
            {
                printf("it should be traversable at (%d, %d)\n", x, z);
                FAIL();
            }

    Waypoint origin(400, 799, angle::deg(0));
    Waypoint goal(400, 0, angle::deg(0));

    // Disable kinematic invalid positions
    f1.processKinematicExclusionAreas(origin, goal);

    std::vector<Waypoint> curve1 = Interpolator::kinematicCurve(f1.width(), f1.height(), origin, angle::deg(-40), 5.34 * 8, -1);
    std::vector<Waypoint> curve2 = Interpolator::kinematicCurve(f1.width(), f1.height(), origin, angle::deg(40), 5.34 * 8, -1);

    goal.set_heading(angle::deg(180).rad());
    std::vector<Waypoint> curve3 = Interpolator::kinematicCurve(f1.width(), f1.height(), goal, angle::deg(-40), 5.34 * 8, -1);
    std::vector<Waypoint> curve4 = Interpolator::kinematicCurve(f1.width(), f1.height(), goal, angle::deg(40), 5.34 * 8, -1);

    std::vector<std::vector<Waypoint>> lst = {curve1, curve2, curve3, curve4};

    std::vector<uchar> dest(static_cast<size_t>(f1.width()) * f1.height() * 3);
    cv::Mat cimg(f1.height(), f1.width(), CV_8UC3, dest.data());

    // for (std::vector<Waypoint> curve : lst)
    // {
    //     for (Waypoint p : curve)
    //     {
    //         cv::Vec3b &pixel = cimg.at<cv::Vec3b>(p.z(), p.x());
    //         pixel[0] = 0;
    //         pixel[1] = 255;
    //         pixel[2] = 0;
    //     }
    // }

    for (int z = 0; z < f1.height(); z++)
    {
        for (auto curve : lst)
        {
            auto [min, max] = minMaxCurve(curve, z);
            if (max == -1 || min > f1.width())
                continue;

            for (int x = 0; x < f1.width(); x++)
            {
                // if (!f1.isTraversable(x, z) && (x < min || x > max))
                if (x >= min && x <= max)
                {
                    if (f1.isTraversable(x, z))
                    {
                        printf("should NOT be traversable at %d,%d\n", x, z);
                        cv::Vec3b &pixel = cimg.at<cv::Vec3b>(z, x);
                        pixel[0] = 128;
                        pixel[1] = 0;
                        pixel[2] = 128;
                    }
                }
            }
        }
    }

    // for (std::vector<Waypoint> curve : lst)
    // {
    //     for (Waypoint p : curve)
    //     {
    //         cv::Vec3b &pixel = cimg.at<cv::Vec3b>(p.z(), p.x());
    //         pixel[0] = 0;
    //         pixel[1] = 255;
    //         pixel[2] = 0;
    //     }
    // }

    cv::imwrite("outp.png", cimg);

    int l = 1;
}