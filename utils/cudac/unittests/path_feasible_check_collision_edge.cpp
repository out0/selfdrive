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

waypoint parseWaypoint(const std::string &str)
{
    waypoint wp;
    sscanf(str.c_str(), "(%d, %d, 0)", &wp.x, &wp.y);
    return wp;
}

// Function to convert JSON array to a vector of waypoints
std::vector<waypoint> parseWaypoints(const json &jsonArray)
{
    std::vector<waypoint> waypoints;
    for (const auto &item : jsonArray)
    {
        waypoints.push_back(parseWaypoint(item));
    }
    return waypoints;
}

float *read_waypoint_list(const json &jsonArray, int &size)
{
    size = jsonArray.size();
    float *points = new float[4 * size];
    float x = 0, z = 0, h = 0;
    int pos = 0;

    for (const std::string &item : jsonArray)
    {
        sscanf(item.c_str(), "(%f, %f, %f)", &x, &z, &h);
        points[pos] = x;
        points[pos + 1] = z;
        points[pos + 2] = h;
        points[pos + 3] = 0;
        pos += 4;
    }

    return points;
}

void assert_all_points_feasible_to_path(CudaFrame &f, const json &jsonArray)
{
    int count = 0;
    float *points = read_waypoint_list(jsonArray, count);
    f.checkFeasibleWaypoints(points, count, true);
    // printf ("not feasible: \n");
    for (int i = 0; i < count; i++)
    {
        if (points[4 * i + 3] == 0)
        {
            int x = (int)points[4 * i];
            int z = (int)points[4 * i + 1];
            float h = (int)points[4 * i + 2];
            printf("Should be a feasible point (%d, %d) heading %f\n", x, z, h);
            FAIL();
        }
    }

    delete[] points;
}
void assert_all_points_feasible_l_c_r(CudaFrame &f, const json &jsonArray)
{
    int c1 = 0, c2 = 0, c3 = 0;
    float *points_l = read_waypoint_list(jsonArray["left"], c1);
    float *points_c = read_waypoint_list(jsonArray["center"], c2);
    float *points_r = read_waypoint_list(jsonArray["right"], c3);

    float *all = new float[4* (c1 + c2 + c3)];

    int pos = 0;
    for (int i = 0; i < c1; i++, pos++)
    {
        for (int j = 0; j < 4; j++)
            all[4 * pos + j] = points_l[4 * i + j];
    }
    for (int i = 0; i < c2; i++, pos++)
    {
        for (int j = 0; j < 4; j++)
            all[4 * pos + j] = points_c[4 * i + j];
    }
    for (int i = 0; i < c3; i++, pos++)
    {
        for (int j = 0; j < 4; j++)
            all[4 * pos + j] = points_r[4 * i + j];
    }

    f.checkFeasibleWaypoints(all, pos, true);
    for (int i = 0; i < pos; i++)
    {
        if (all[4 * i + 3] == 0)
        {
            int x = (int)all[4 * i];
            int z = (int)all[4 * i + 1];
            float h = (int)all[4 * i + 2];
            printf("Should be a feasible point (%d, %d) heading %f\n", x, z, h);
            FAIL();
        }
    }

    delete points_l;
    delete points_c;
    delete points_r;
    delete[] all;    
}

TEST(PathFeasibleCheck, TestEgdeCollision)
{
    // cv::Mat m = cv::imread("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/cudac/unittests/test_data/bev_1.png", cv::IMREAD_COLOR);
    cv::Mat m = cv::imread("test_data/bev_1.png", cv::IMREAD_COLOR);
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

    std::string data;

    std::ifstream inputFile("test_data/collision_path_1.log");
    ASSERT_TRUE(inputFile.is_open());
    json j;
    inputFile >> j;
    inputFile.close();

    assert_all_points_feasible_to_path(f, j["left"]);
    assert_all_points_feasible_to_path(f, j["center"]);
    assert_all_points_feasible_to_path(f, j["right"]);
    assert_all_points_feasible_l_c_r(f, j);

}

