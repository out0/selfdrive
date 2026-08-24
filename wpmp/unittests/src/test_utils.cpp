#include <cmath>
#include <stdio.h>
#include <fstream>
#include "test_utils.h"
#include "../../src/wpmp_data.h"

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

// #ifdef DRIVELESS_CUDA_ENABLED
// #else
// std::unique_ptr<float3[]> createEmptySearchFrame(int width, int height)
// {
//     auto ptr = std::make_unique<float3[]>(width * height);
//     long size = height * width;
//     for (int i = 0; i < size; i++)
//     {
//         ptr.get()[i].x = 0;
//         ptr.get()[i].y = 0;
//         ptr.get()[i].z = 0;
//     }
//     return ptr;
// }

// #endif

// SearchFrame *createEmptySearchFrame(int width, int height)
// {
//     auto ptr = new SearchFrame(width, height, {-1, -1}, {-1, -1});
//     long size = height * width;
//     float *p = new float[size * 3];

//     for (int i = 0; i < size; i++)
//     {
//         int pos = 3 * i;
//         p[pos] = 0;
//         p[pos + 1] = 0;
//         p[pos + 2] = 0;
//     }
//     ptr->copyFrom(p);
//     delete[] p;
//     return ptr;
// }

SearchFrame *createEmptySearchFrame(
    int width,
    int height,
    std::pair<int, int> lower_bound,
    std::pair<int, int> upper_bound,
    std::pair<int, int> sz_dim)
{
    SearchFrame *f = new SearchFrame(width, height, lower_bound, upper_bound, sz_dim);
    f->setClassCosts({1, -1});
    f->setClassColors({{0, 0, 0},
                       {255, 255, 255}});

    float *ptr = new float[width * height * 3];
    memset(ptr, 0x0, sizeof(float) * width * height * 3);

    f->copyFrom(ptr);

    delete[] ptr;
    return f;
}

void showSearchParameters(SearchFrame *frame)
{
    int *params = frame->getFrameParamsPtr();
    float *phys_params = frame->getPhysicalParamsPtr();

    printf("FRAME_PARAM_WIDTH: %d\n", params[FRAME_PARAM_WIDTH]);
    printf("FRAME_PARAM_HEIGHT: %d\n", params[FRAME_PARAM_HEIGHT]);
    printf("FRAME_PARAM_MIN_DIST_X: %d\n", params[FRAME_PARAM_MIN_DIST_X]);
    printf("FRAME_PARAM_MIN_DIST_Z: %d\n", params[FRAME_PARAM_MIN_DIST_Z]);
    printf("FRAME_PARAM_LOWER_BOUND_X: %d\n", params[FRAME_PARAM_LOWER_BOUND_X]);
    printf("FRAME_PARAM_LOWER_BOUND_Z: %d\n", params[FRAME_PARAM_LOWER_BOUND_Z]);
    printf("FRAME_PARAM_UPPER_BOUND_X: %d\n", params[FRAME_PARAM_UPPER_BOUND_X]);
    printf("FRAME_PARAM_UPPER_BOUND_Z: %d\n", params[FRAME_PARAM_UPPER_BOUND_Z]);
    printf("FRAME_PARAM_CENTER_X: %d\n", params[FRAME_PARAM_CENTER_X]);
    printf("FRAME_PARAM_CENTER_Z: %d\n", params[FRAME_PARAM_CENTER_Z]);
    printf("FRAME_SAFE_ZONE_CHECKED: %d\n", frame->isSafeZoneChecked());
    printf("FRAME_SAFE_ZONE_VECTORIAL_CHECKED: %d\n", frame->isVectorialSafeZoneChecked());
    printf("FRAME_DISTANCE_TO_GOAL_PROCESSED: %d\n", frame->isDistanceToGoalProcessed());
    printf("FRAME_PREPROCESS_DIST_TO_GOAL_ENABLED: %d\n", params[FRAME_PREPROCESS_DIST_TO_GOAL_ENABLED]);
    printf("FRAME_SEARCH_ZONE_DIM_WIDTH: %d\n", params[FRAME_SEARCH_ZONE_DIM_WIDTH]);
    printf("FRAME_SEARCH_ZONE_DIM_HEIGHT: %d\n", params[FRAME_SEARCH_ZONE_DIM_HEIGHT]);
    printf("FRAME_SEARCH_ZONE_GRID_WIDTH: %d\n", params[FRAME_SEARCH_ZONE_GRID_WIDTH]);
    printf("FRAME_SEARCH_ZONE_GRID_HEIGHT: %d\n", params[FRAME_SEARCH_ZONE_GRID_HEIGHT]);

    printf ("--------------------\n");

    printf("PHYSICAL_PARAM_REAL_WIDTH: %f\n", phys_params[PHYSICAL_PARAM_REAL_WIDTH]);
    printf("PHYSICAL_PARAM_REAL_HEIGHT: %f\n", phys_params[PHYSICAL_PARAM_REAL_HEIGHT]);

    printf("PHYSICAL_PARAM_RATE_W: %f\n", phys_params[PHYSICAL_PARAM_RATE_W]);
    printf("PHYSICAL_PARAM_RATE_H: %f\n", phys_params[PHYSICAL_PARAM_RATE_H]);
    printf("PHYSICAL_PARAM_RATE_SQUARE: %f\n", phys_params[PHYSICAL_PARAM_RATE_SQUARE]);

    printf("PHYSICAL_PARAM_INV_RATE_W: %f\n", phys_params[PHYSICAL_PARAM_INV_RATE_W]);
    printf("PHYSICAL_PARAM_INV_RATE_H: %f\n", phys_params[PHYSICAL_PARAM_INV_RATE_H]);
    printf("PHYSICAL_PARAM_INV_RATE_SQUARE: %f\n", phys_params[PHYSICAL_PARAM_INV_RATE_SQUARE]);

    printf("PHYSICAL_PARAM_MAX_STEERING_RAD: %f\n", phys_params[PHYSICAL_PARAM_MAX_STEERING_RAD]);
    printf("PHYSICAL_PARAM_MAX_STEERING_DEG: %f\n", phys_params[PHYSICAL_PARAM_MAX_STEERING_DEG]);

    printf("PHYSICAL_PARAM_WHEELBASE: %f\n", phys_params[PHYSICAL_PARAM_WHEELBASE]);
    printf("PHYSICAL_PARAM_WHEELBASE_PX: %f\n", phys_params[PHYSICAL_PARAM_WHEELBASE_PX]);

    printf("PHYSICAL_PARAM_MAX_CURVATURE: %f\n", phys_params[PHYSICAL_PARAM_MAX_CURVATURE]);
}

void assertInt2Equal(int2 a, int2 b)
{
    if (a.x != b.x || a.y != b.y)
    {
        printf("(%d, %d) != (%d, %d)\n", a.x, a.y, b.x, b.y);
        FAIL();
    }
}

SearchFrame *buildTestSearchFrame()
{
    SearchFrame *f = new SearchFrame(256, 256, {-1, -1}, {-1, -1});
    std::vector<float> costs = {
        {0},
        {1},
        {2},
        {3},
        {4},
        {-1}};

    float *ptr = new float[256 * 256 * 3];
    for (int i = 0; i < 256 * 256 * 3; i++)
        ptr[i] = 0;

    f->setClassCosts(costs);
    f->copyFrom(ptr);
    delete[] ptr;
    return f;
}

cv::Mat exportGraph(SearchFrame *frame, WGraph *graph, const std::string &file)
{
    std::vector<uchar> dest(static_cast<size_t>(frame->width()) * frame->height() * 3);
    frame->exportToColorFrame(dest.data());
    cv::Mat cimg(frame->height(), frame->width(), CV_8UC3, dest.data());

    auto frame_conf = graph->get_node_conf();

    for (int h = 0; h < frame->height(); h++)
        for (int w = 0; w < frame->width(); w++)
        {
            int pos = h * frame->width() + w;

            int type = NODE_TYPE(frame_conf->getPtr(), pos);

            cv::Vec3b &pixel = cimg.at<cv::Vec3b>(h, w);

            switch (type)
            {
            case NODE_TYPE_NULL_CONNECTED_TO_GOAL:
                pixel[0] = 0;
                pixel[1] = 255;
                pixel[2] = 0;
                break;
            case NODE_TYPE_GRAPH_CONNECTED_TO_GOAL:
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 255;
                break;
            }
        }
    if (!file.empty())
        cv::imwrite(file, cimg);
    return cimg;
}