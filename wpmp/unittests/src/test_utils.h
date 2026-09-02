#ifndef __TEST_UTILS_DRIVELESS_H
#define __TEST_UTILS_DRIVELESS_H

#include <driveless/cuda_basic.h>
#include <driveless/search_frame.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include "../../include/wpmp_graph.h"

extern bool _ASSERT_DEQ(double a, double b, int tolerance = 4);
#define ASSERT_DEQ(a, b) ASSERT_TRUE(_ASSERT_DEQ(a, b))

SearchFrame *createEmptySearchFrame(
    int width,
    int height,
    std::pair<int, int> lower_bound,
    std::pair<int, int> upper_bound,
    std::pair<int, int> sz_dim = {-1, -1}
);

SearchFrame *createEmptySearchFramePtr(int width, int height);

void assertInt2Equal(int2 a, int2 b);

SearchFrame *buildTestSearchFrame();

cv::Mat exportGraph(SearchFrame *frame, WGraph *graph, const std::string &file);

void showSearchParameters(SearchFrame * frame);

template <typename Func>
auto timeIt(std::string name, Func&& f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "[" << name << "]" << " execution time: " << duration / 1000 << " ms" << " (" << duration << ") us" << std::endl;
}

#endif