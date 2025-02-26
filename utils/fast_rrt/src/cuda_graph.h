#pragma once

#include "cuda_basic.h"
#include <vector>

class CudaGraph
{
    double4 *graph;
    double *graph_cost;
    std::vector<int2> _unordered_nodes;
    int3 *point;
    int *classCosts;
    double *checkParams;
    unsigned int *pcount;
    long long *bestValue;
    int width;
    int height;
    double3 _center;
    double _goal_heading_deg;
    int _max_steering_angle_deg;

    int __random_gen(int min, int max);
    void __expandNodesCPU(float3 *og, int max_step_size);
    void __expandNodesGPU(float3 *og, int max_step_size);

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
        double _lr);

    ~CudaGraph();

    // Basic stuff
    void clear();
    double getHeading(int x, int z);
    void add(int x, int z, double heading, int parent_x, int parent_z, double cost, bool temporary_node = false);
    void remove(int x, int z);
    double3 getParent(int x, int z);
    void setParent(int x, int z, int parent_x, int parent_z);
    bool checkInGraph(int x, int z);
    void setCost(int x, int z, double cost);
    double getCost(int x, int z);
    unsigned int count();
    
    // CUDA listing removed [deprecated because does not support being called many times. Cuda doesnt handle well alloc/free]
    //void list(double *result, int count);
    int2 get_random_node();
    std::vector<int2>& list();

    void setVelocity(double velocity_meters_per_s);
    void setGoalHeading(double heading);

    // Test stuff
    void drawNodes(float3 *og);
    void drawKinematicPath(float3 *og, double3 &start, double3 &end);

    // RRT operations
    // ------------------------------------------------------------------------------------

    // connects (parent_x, parent_z) to (x', z') which is the kinematic-closest point from x,z, if feasible
    bool connectToGraph(float3 *og, int parent_x, int parent_z, int x, int z);

    // connects (parent_x, parent_z) to (x', z') which is the kinematic-closest point from x,z, if feasible
    int2 deriveNode(float3 *og, int parent_x, int parent_z, double angle_deg, double size);

    // checks that the connection between start and end is feasible
    bool checkConnectionFeasible(float3 *og, double3 &start, double3 end);

    /// Returns the nearest neighbor, based on euclidean distance only
    int2 find_nearest_neighbor(int x, int z);

    // Returns the best neighbor based on cost
    int2 find_best_neighbor(int x, int z, float radius);

    /// Returns the nearest neighbor p, based on euclidean distance, which can draw a feasible p -> (x,z) .
    int2 find_nearest_feasible_neighbor(float3 *og, int x, int z);

    // Returns the best neighbor based on cost that can reach x,z
    int2 find_best_feasible_neighbor(float3 *og, int x, int z, float radius);

    // orders nodes within a radius to verify if they should use x,z as their parent node.
    void optimizeGraphWithNode(float3 *og, int x, int z, float radius);


    // Full parallel RRT (experimental)
    // connects (parent_x, parent_z) to (x', z') which is the kinematic-closest point from x,z, if feasible
    void expandNodes(float3 *og, int max_step_size);
};