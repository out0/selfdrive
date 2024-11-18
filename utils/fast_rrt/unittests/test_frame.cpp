#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "../src/cuda_frame.h"
#include "../src/cuda_graph.h"
#include "test_utils.h"
#include "test_frame.h"
#include "../src/kinematic_model.h"
#include <unordered_set>
#include <math.h>

#define OG_REAL_WIDTH 34.641016151377535
#define OG_REAL_HEIGHT 34.641016151377535
#define MAX_STEERING_ANGLE 40

#include "test_frame.h"

int TestFrame::toSetKey(int x, int z)
{
    return 1000 * x + z;
}

void TestFrame::drawNode(CudaGraph *graph, float3 *imgPtr, int x, int z, double heading)
{
    double3 parent = graph->getParent(x, z);

    if (parent.x < 0 or parent.y < 0)
        return;

    double3 start, end;
    start.x = parent.x;
    start.y = parent.y;
    start.z = parent.z;
    end.x = x;
    end.y = z;
    end.z = heading;

    graph->drawKinematicPath(imgPtr, start, end);
}

void change_color(int3 *imgPtr, int width, int x, int z, int r, int g, int b)
{
    int pos = z * width + x;
    imgPtr[pos].x = r;
    imgPtr[pos].y = g;
    imgPtr[pos].z = b;
}

void debug_draw_point(int3 *imgPtr, int width, int x, int z)
{
    for (int i = -1; i <= 1; i++)
    {
        change_color(imgPtr, width, x + i, z, 255, 0, 0);
        change_color(imgPtr, width, x, z + i, 255, 0, 0);
    }
}

void draw_obstacles(float3 *imgPtr, CudaFrame *og)
{

    for (int z = 0; z < og->getHeight(); z++)
    {
        for (int x = 0; x < og->getWidth(); x++)
        {
            int pos = z * og->getWidth() + x;
            if (static_cast<int>(round(imgPtr[pos].x)) == 28)
            {
                imgPtr[pos].x = 0;
                imgPtr[pos].y = 255;
                imgPtr[pos].z = 0;
            }
        }
    }
}
void draw_obstacles(int3 *imgPtr, CudaFrame *og)
{

    for (int z = 0; z < og->getHeight(); z++)
    {
        for (int x = 0; x < og->getWidth(); x++)
        {
            int pos = z * og->getWidth() + x;
            if (static_cast<int>(round(imgPtr[pos].x)) == 28)
            {
                imgPtr[pos].x = 0;
                imgPtr[pos].y = 255;
                imgPtr[pos].z = 0;
            }
        }
    }
}

void TestFrame::drawNodeDebug(CudaGraph *graph, float3 *imgPtr, int3 *imgPtrOutput, int x, int z, double heading)
{
    double3 parent = graph->getParent(x, z);
    int width = OG_WIDTH;
    int height = OG_HEIGHT;
    int min_dist_x = MIN_DIST_X / 2;
    int min_dist_z = MIN_DIST_Z / 2;
    // int lower_bound_ego_x = LOWER_BOUND_X;
    // int lower_bound_ego_z = LOWER_BOUND_Z;
    // int upper_bound_ego_x = UPPER_BOUND_X;
    // int upper_bound_ego_z = UPPER_BOUND_Z;
    int lower_bound_ego_x = -1;
    int lower_bound_ego_z = -1;
    int upper_bound_ego_x = -1;
    int upper_bound_ego_z = -1;

    if (parent.x < 0 or parent.y < 0)
        return;

    double3 start, end;
    start.x = parent.x;
    start.y = parent.y;
    start.z = parent.z;
    end.x = x;
    end.y = z;
    end.z = heading;

    double3 center = {128, 128, 0};

    debug_draw_point(imgPtrOutput, width, x, z);

    std::vector<double3> curve = CurveGenerator::buildCurveWaypoints(center, _rw, _rh, _lr, MAX_STEERING_ANGLE, start, end, 1.0, false);

    for (auto p : curve)
    {
        change_color(imgPtrOutput, width, p.x, p.y, 255, 255, 255);

        double c = cos(p.z);
        double s = sin(p.z);
        int max_size = width * height;

        for (int i = -min_dist_z; i <= min_dist_z; i++)
            for (int j = -min_dist_x; j <= min_dist_x; j++)
            {
                int xl = static_cast<int>(j * c - i * s + x);
                int zl = static_cast<int>(j * s + i * c + z);

                if (xl < 0 || xl >= width)
                    continue;

                if (zl < 0 || zl >= height)
                    continue;

                if (xl >= lower_bound_ego_x && xl <= upper_bound_ego_x && zl >= upper_bound_ego_z && zl <= lower_bound_ego_z)
                    continue;

                int pos = zl * width + xl;
                if (pos >= max_size || pos < 0)
                    continue;

                int segmentation_class = static_cast<int>(imgPtr[pos].x);

                if (segmentation_class == 0 || segmentation_class == 28)
                {
                    change_color(imgPtrOutput, width, xl, zl, 0, 0, 255);
                }
                else
                {
                    change_color(imgPtrOutput, width, xl, zl, 0, 255, 255);
                }
            }
    }
}

TestFrame::TestFrame(int default_fill_value)
{
    og = create_default_cuda_frame(default_fill_value);

    _rw = OG_WIDTH / OG_REAL_WIDTH;
    _rh = OG_HEIGHT / OG_REAL_HEIGHT;
    _lr = 0.5 * (LOWER_BOUND_Z - UPPER_BOUND_Z) / _rh;

    graph = new CudaGraph(
        OG_WIDTH,
        OG_HEIGHT,
        MIN_DIST_X,
        MIN_DIST_Z,
        LOWER_BOUND_X,
        LOWER_BOUND_Z,
        UPPER_BOUND_X,
        UPPER_BOUND_Z,
        _rw, _rh,
        MAX_STEERING_ANGLE,
        _lr, 1.0);
}

TestFrame::~TestFrame()
{
    delete og;
    delete graph;
}

void TestFrame::toFile(const char *filename = "dump.png")
{
    dump_cuda_frame_to_file(og, filename);
}
CudaGraph *TestFrame::getGraph()
{
    return graph;
}
CudaFrame *TestFrame::getCudaGrame()
{
    return og;
}

void TestFrame::addArea(int x1, int z1, int x2, int z2, int classType)
{
    float3 *imgPtr = og->getFramePtr();

    for (int z = std::min(z1, z2); z < std::max(z1, z2); z++)
    {
        for (int x = std::min(x1, x2); x < std::max(x1, x2); x++)
        {
            imgPtr[z * og->getWidth() + x].x = classType;
        }
    }
}

void TestFrame::drawGraph()
{
    float3 *imgPtr = og->getFramePtr();

    int num_nodes = graph->count();
    if (num_nodes == 0)
        return;

    double *nodes = new double[6 * sizeof(double) * num_nodes];
    graph->list(nodes, num_nodes);

    std::unordered_set<int> drawn;

    for (int i = 0; i < num_nodes; i++)
    {
        int pos = 6 * i;
        int x = static_cast<int>(nodes[pos]);
        int z = static_cast<int>(nodes[pos + 1]);
        int key = toSetKey(x, z);

        if (drawn.find(key) != drawn.end())
            continue;

        double heading = static_cast<int>(nodes[pos + 2]);
        drawNode(graph, imgPtr, x, z, heading);
        drawn.insert(key);
    }

    graph->drawNodes(imgPtr);
    draw_obstacles(imgPtr, og);
}

void TestFrame::drawGraphDebugTo(const char *filename)
{
    float3 *imgPtr = og->getFramePtr();

    int3 *output = new int3[og->getWidth() * og->getHeight()];

    int num_nodes = graph->count();
    if (num_nodes == 0)
        return;

    double *nodes = new double[6 * sizeof(double) * num_nodes];
    graph->list(nodes, num_nodes);

    std::unordered_set<int> drawn;

    for (int i = 0; i < num_nodes; i++)
    {
        int pos = 6 * i;
        int x = static_cast<int>(nodes[pos]);
        int z = static_cast<int>(nodes[pos + 1]);
        int key = toSetKey(x, z);

        if (drawn.find(key) != drawn.end())
            continue;

        double heading = static_cast<int>(nodes[pos + 2]);
        drawNodeDebug(graph, imgPtr, output, x, z, heading);
        drawn.insert(key);
    }

    draw_obstacles(output, og);

    cv::Mat cimg = cv::Mat(og->getHeight(), og->getWidth(), CV_8UC3);

    for (int i = 0; i < og->getHeight(); i++)
        for (int j = 0; j < og->getWidth(); j++)
        {
            cv::Vec3b &pixel = cimg.at<cv::Vec3b>(i, j);
            int idx = i * 256 + j;

            // Set the channels: Blue, Green, Red
            pixel[0] = output[idx].x; // Blue
            pixel[1] = output[idx].y; // Green
            pixel[2] = output[idx].z; // Red
        }

    cv::imwrite(filename, cimg);
}

void TestFrame::dump_cuda_frame_to_file(CudaFrame *frame, const char *filename)
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