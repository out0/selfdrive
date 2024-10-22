#pragma once

#include "cuda_basic.h"


class CudaGraph
{
    float4 *frame;
    int3 *point;
    int width;
    int height;

public:
    CudaGraph(int width, int height);
    ~CudaGraph();

    void clear();
    int * find_best_neighbor(int x, int z, float radius);
    void add_point(int x, int z, int parent_x, int parent_z, float cost);
    unsigned int count();
    bool checkInGraph(int x, int z);
};