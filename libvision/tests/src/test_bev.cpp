#include <gtest/gtest.h>
#include "test_utils.h"
#include <cmath>
#include <tuple>
#include <driveless/search_frame.h>
#include <driveless/cuda_basic.h>
#include "../../include/bev.h"

TEST(TestBEV, TestBuildBev)
{
    SearchFrame front(512, 512, {-1, -1}, {-1, -1});
    SearchFrame back(512, 512, {-1, -1}, {-1, -1});
    SearchFrame left(512, 512, {-1, -1}, {-1, -1});
    SearchFrame right(512, 512, {-1, -1}, {-1, -1});

    int size = 512*512*3;
    float *empty = new float[size];
    memset(empty, 0x0, size);

    front.copyFrom(empty);
    back.copyFrom(empty);
    left.copyFrom(empty);
    right.copyFrom(empty);

    BEV bev(512, 512, {50, 150}, 1);

    bev.compute(&front, &back, &left, &right);

    SearchFrame *result = bev.get();

    std::vector<std::tuple<int, int, int>> colors = {
        {0, 0, 0},
        {255, 0, 0},
    };

    result->setClassColors(colors);

    exportSearchFrameToFile(*result, "teste.png");

    delete []empty;
    
}