#ifndef __TEST_UTILS_DRIVELESS_H
#define __TEST_UTILS_DRIVELESS_H

#include <driveless/cuda_basic.h>
#include <driveless/search_frame.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "../../include/wpmp_graph.h"

extern bool _ASSERT_DEQ(double a, double b, int tolerance = 4);
#define ASSERT_DEQ(a, b) ASSERT_TRUE(_ASSERT_DEQ(a, b))

SearchFrame *createEmptySearchFrame(
    int width,
    int height,
    std::pair<int, int> lower_bound,
    std::pair<int, int> upper_bound);

SearchFrame *createEmptySearchFramePtr(int width, int height);

void assertInt2Equal(int2 a, int2 b);

SearchFrame *buildTestSearchFrame();

cv::Mat exportGraph(SearchFrame *frame, WGraph *graph, const std::string &file);

#endif