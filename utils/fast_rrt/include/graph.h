#pragma once

#ifndef __GRAPH_DRIVELESS_H
#define __GRAPH_DRIVELESS_H

#include "cuda_grid.h"
#include "../../cudac/include/cuda_frame.h"
#include "angle.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <memory>


#define GRAPH_TYPE_NULL 0
#define GRAPH_TYPE_NODE 1
#define GRAPH_TYPE_TEMP 2
#define GRAPH_TYPE_PROCESSING 3

#define THREADS_IN_BLOCK 256

typedef float3 pose;

class CudaGraph
{
private:
    std::shared_ptr<CudaGrid<int4>> _frame;
    std::shared_ptr<CudaGrid<float3>> _frameData;
    bool __checkLimits(int x, int z);
    unsigned int *_parallelCount = 0;
    bool *_newNodesAdded;
    int2 _gridCenter;
    double *_physicalParams;
    int *_searchSpaceParams;
    float *_classCosts;
    void __initializeRandomGenerator();
    curandState *_randState;
    std::pair<int2 *, int> __listNodes(int type);
    std::pair<int3 *, int> __listAllNodes();
    
    
    unsigned int __countInRange(int xp, int zp, float radius_sqr);
    std::pair<int2 *, int> __listNodesInRange(int type, int x, int z, float radius);

    void __optimizeGraph(float3 *og, int x, int z, float radius, float velocity_m_s, angle goalHeading);
    bool *_goalReached;

protected:
    /// @brief Checks and accepts a derivated path from a node for feasibility. All nodes will be cleared. If acceptable, the last node will be a node in the graph.
    /// @param graph
    /// @param graphData
    /// @param cudaFrame
    /// @param params
    /// @param classCost
    /// @param start
    /// @param lastNode
    /// @return true for accepted path, false otherwise
    bool __checkDerivedPath(float3 *og, int2 start, int2 lastNode);

    /// @brief Checks and accepts all derivated paths for feasibility.
    /// @param searchFrame
    void __checkDerivedPath(float3 *og);

    

public:
    CudaGraph(int width, int height);
    ~CudaGraph();

    void computeRepulsiveFieldAPF(float3 *og, float Kr, int radius);
    void computeAttractiveFieldAPF(float3 *og, float Ka, std::pair<int, int> goal);


    void setSearchParams(std::pair<int, int> minDistance, std::pair<int, int> lowerBound, std::pair<int, int> upperBound);
    void setPhysicalParams(float perceptionWidthSize_m, float perceptionHeightSize_m, angle maxSteeringAngle, float vehicleLength);
    void setClassCosts(const int *costs, int size);
    void add(int x, int z, angle heading, int parent_x, int parent_z, float cost);
    void addTemporary(int x, int z, angle heading, int parent_x, int parent_z, float cost);
    void addStart(int x, int z, angle heading);
    void remove(int x, int z);
    void clear();
    std::vector<int2> list();
    std::vector<int3> listAll();
    std::vector<int2> listInRange(int x, int z, float radius);
    unsigned int count(int type = GRAPH_TYPE_NODE);
    unsigned int countAll();

    inline int height()
    {
        return _frame->height();
    }
    inline int width()
    {
        return _frame->width();
    }
    std::shared_ptr<CudaGrid<int4>> getFramePtr()
    {
        return _frame;
    }
    std::shared_ptr<CudaGrid<float3>> getFrameDataPtr()
    {
        return _frameData;
    }

    double * getPhysicalParams() {
        return _physicalParams;
    }

    int2 getCenter() {
        return _gridCenter;
    }

    bool checkInGraph(int x, int z);
    void setParent(int x, int z, int parent_x, int parent_z);
    int2 getParent(int x, int z);
    angle getHeading(int x, int z);
    void setHeading(int x, int z, angle heading);
    float getCost(int x, int z);
    void setCost(int x, int z, float cost);

    void setType(int x, int z, int type);

    int getType(int x, int z);

    /// @brief Derivates a node on position {x, z} for the specified steeringAngle, pathSize and velocity_m_s. The node must exist in the graph.
    /// @param x
    /// @param z
    /// @param heading
    /// @return final node of the path
    int2 derivateNode(float3 *og, angle goalHeading, angle steeringAngle, double pathSize, float velocity_m_s, int x, int z);
    
    /// @brief Derivates all nodes in graph with a random steering angle and pathSize, for the specified maxSteeringAngle, maxPathSize, and velocity_m_s.
    /// @param maxSteeringAngle
    /// @param maxPathSize
    /// @param velocity_m_s
    void expandTree(float3 *og, angle goalHeading, float maxPathSize, float velocity_m_s, bool frontierExpansion);

    /// @brief Accepts a derivated node and connects it to the graph.
    /// @param start
    /// @param lastNode
    /// @return true for accepted nodes, false otherwise
    void acceptDerivedNode(int2 start, int2 lastNode);

    /// @brief Accepts all derivated nodes and connects them to the graph.
    /// @return
    void acceptDerivedNodes();

    /// @brief Finds the best node in graph (with the lowest cost) that is feasible with the given heading, in a given search radius
    /// @param searchFrame
    /// @param radius
    /// @param x
    /// @param z
    /// @param heading
    /// @return
    int2 findBestNode(float3 *og, angle heading, float radius, int x, int z);


    /// @brief Checks if there is a feasible connection between start and end, at the given velocity and max steering angle
    /// @param searchFrame 
    /// @param start 
    /// @param end 
    /// @param velocity_m_s 
    /// @param maxSteeringAngle 
    /// @return 
    bool checkFeasibleConnection(float3 *og, int2 start, int2 end, int velocity_m_s);

    /// @brief Returns true if any node in the graph is at a distance equals or lower than distanceToGoalTolerance and is feasible on the given heading.
    /// @param searchFrame 
    /// @param goal 
    /// @param heading 
    /// @param distanceToGoalTolerance 
    /// @return 
    bool checkGoalReached(float3 *og, int2 goal, angle heading, float distanceToGoalTolerance);


    // /// @brief Optimizes the graph with the added new nodes, changing node parents for total cost reduction (RRT*)
    // /// @param searchFrame
    // /// @param radius
    void optimizeGraph(float3 *og, angle goalHeading, float radius, float velocity_m_s);

    void optimizeNode(float3 *og, int x, int z, float radius, float velocity_m_s, int numNodesInGraph);

    void dumpGraph(const char *filename);

    void readfromDump(const char *filename);

    bool checkNewNodesAddedOnTreeExpansion();
};

#endif