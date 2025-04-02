#include <cmath>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "test_utils.h"
#include "../../include/graph.h"

bool _ASSERT_DEQ(double a, double b, int tolerance)
{
    double p = pow(10, -tolerance);

    if (abs(a - b) > p)
    {
        printf("ASSERT_DEQ failed: %f != %f, tolerance: %f\n", a, b, p);
        return false;
    }

    return true;
}


std::vector<int2> get_planned_path(CudaGraph *graph, float3 *ptr, angle goal_heading, int goal_x, int goal_z, float distToGoalTolerance)
{
    // res.push_back(*_goal);
    int2 n = graph->findBestNode(ptr, goal_heading, distToGoalTolerance, goal_x, goal_z);
    std::vector<int2> res;

    while (n.x != -1 && n.y != -1)
    {
        res.push_back({n.x, n.y});
        n = graph->getParent(n.x, n.y);
    }

    std::reverse(res.begin(), res.end());
    return res;
}

void exportGraph(CudaGraph *graph, const char *filename, std::vector<int2> *path)
{
    cv::Mat cimg = cv::Mat(graph->height(), graph->width(), CV_8UC3, cv::Scalar(0));


    int3 *ptr = graph->getFramePtr()->getCudaPtr();

    for (int h = 0; h < graph->height(); h++)
        for (int w = 0; w < graph->width(); w++)
        {
            long pos = h * graph->width() + w;
            if (ptr[pos].z == 0)
                continue;

            cv::Vec3b &pixel = cimg.at<cv::Vec3b>(h, w);

            switch (ptr[pos].z)
            {
            case GRAPH_TYPE_NODE:
                pixel[0] = 255;
                pixel[1] = 255;
                pixel[2] = 255;
                break;
            case GRAPH_TYPE_TEMP:
                pixel[0] = 0;
                pixel[1] = 255;
                pixel[2] = 0;
            case GRAPH_TYPE_PROCESSING:
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 255;
            default:
                break;
            }
        }

        if (path != nullptr) {
            for (auto p : *path) {
                cv::Vec3b &pixel = cimg.at<cv::Vec3b>(p.y, p.x);
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 255;
            }
        }

    cv::imwrite(filename, cimg);
}

float3 *createEmptySearchFrame(int width, int height)
{
    float3 *ptr;
    cudaAllocMapped(&ptr, sizeof(float3) * width * height);
    long size = height * width;
    for (int i = 0; i < size; i++)
    {
        ptr[i].x = 0;
        ptr[i].y = 0;
        ptr[i].z = 0;
    }
    return ptr;
}

void destroySearchFrame(float3 * ptr) {
    cudaFree(ptr);
}