#pragma once

#include "cuda_basic.h"
#include <vector>

class CudaGraph
{
    double4 *graph;
    double *graph_cost;
    int3 *point;
    int *classCosts;
    double *checkParams;
    unsigned int *pcount;
    long long *bestValue;
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
        double _rate_w,
        double _rate_h,
        double _max_steering_angle_deg,
        double _lr,
        double velocity_meters_per_s);

    ~CudaGraph();

    // Basic stuff
    void clear();
    void add(int x, int z, double heading, int parent_x, int parent_z, double cost);
    void remove(int x, int z);
    double3 getParent(int x, int z);
    void setParent(int x, int z, int parent_x, int parent_z);
    bool checkInGraph(int x, int z);
    void setCost(int x, int z, double cost);
    double getCost(int x, int z);
    unsigned int count();
    void list(double *result, int count);

    // Test stuff
    void drawNodes(float3 *og);
    void drawKinematicPath(float3 *og, double3 &start, double3 &end);

    // RRT operations
    // ------------------------------------------------------------------------------------

    // checks that the connection between start and end is feasible
    bool checkConnectionFeasible(float3 *og, double3 &start, double3 end);
    /// Returns the nearest neighbor, based on euclidean distance only
    int2 find_nearest_neighbor(int x, int z);
    // Returns the best neighbor based on cost
    int2 find_best_neighbor(int x, int z, float radius);   //NEEDS UNITTEST


    /// Returns the nearest neighbor p, based on euclidean distance, which can draw a feasible p -> (x,z) .
    int2 find_nearest_feasible_neighbor(float3 *og, int x, int z);
    // Returns the best neighbor based on cost that can reach x,z
    int2 find_best_feasible_neighbor(float3 *og, int x, int z, float radius);   //NEEDS UNITTEST

    // orders nodes within a radius to verify if they should use x,z as their parent node.
    void optimizeGraphWithNode(float3 *og, int x, int z, float radius);

};