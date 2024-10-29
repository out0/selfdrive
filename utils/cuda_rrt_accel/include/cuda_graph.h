#pragma once

#include "cuda_basic.h"

class CudaGraph
{
    float4 *frame;
    int3 *point;
    int *classCosts;
    int width;
    int height;
    int min_dist_x;
    int min_dist_z;
    int lower_bound_ego_x;
    int lower_bound_ego_z;
    int upper_bound_ego_x;
    int upper_bound_ego_z;

public:
    CudaGraph(int width, int height);

    ~CudaGraph();

    void clear();
    int *find_nearest_neighbor(int x, int z);
    int *find_best_neighbor(int x, int z, float radius);
    void add_point(int x, int z, int parent_x, int parent_z, float cost);

    unsigned int count();
    bool checkInGraph(int x, int z);
    int *getParent(int x, int z);
    void listNodes(float *res, int count);
    float getCost(int x, int z);
    // int listGraphPoints(void *self, int *points);
};