#pragma once

#include "cuda_basic.h"

class CudaGraph
{
    float4 *graph;
    int3 *point;
    int *classCosts;
    float *checkParams;
    unsigned int *pcount;
    int *bestValue;
    int width;
    int height;

public:
    CudaGraph(
        int width,
        int height,
        int min_dist_x,
        int min_dist_z,
        int lower_bound_ego_x,
        int lower_bound_ego_z,
        int upper_bound_ego_x,
        int upper_bound_ego_z,
        float _rate_w,
        float _rate_h,
        float _max_steering_angle_deg,
        float _lr,
        float velocity_meters_per_s);

    ~CudaGraph();

    // Basic stuff
    void clear();
    void add(int x, int z, int parent_x, int parent_z, float cost);
    unsigned int count();
    bool checkInGraph(int x, int z);
    int2 getParent(int x, int z);
    float getCost(int x, int z);

    // Test stuff
    void drawKinematicPath(float3 *og, float3 &start, float3 &end);

    bool checkConnectionFeasible(float3 *frame, float3 &start, float3 end);

    // RRT operations
    //int2 find_nearest_feasible_neighbor(int x, int z);
};