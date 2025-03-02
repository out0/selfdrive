#pragma once

#ifndef __FASTRRT_DRIVELESS_H
#define __FASTRRT_DRIVELESS_H

#include <cmath>
#include <chrono>
#include "graph.h"
#include "waypoint.h"
#include "../../cudac/include/cuda_frame.h"
#include "../../cudac/include/class_def.h"
#include <vector>

typedef float3* cudaPtr;

class FastRRT
{
private:
    CudaGraph _graph;
    std::chrono::time_point<std::chrono::high_resolution_clock> _exec_start;
    bool _goal_found;
    long _timeout_ms;
    float _maxPathSize;
    float _distToGoalTolerance;
    Waypoint *_goal;
    cudaPtr _ptr;
    float _planningVelocity_m_s;
    int2 _bestNode;

    void __set_exec_started();
    long __get_exec_time_ms();
    bool __check_timeout();
    void __clean_search_graph();

public:
    FastRRT(int width, int height,
            float perceptionWidthSize_m,
            float perceptionHeightSize_m,
            angle maxSteeringAngle,
            float vehicleLength,
            int timeout_ms,
            std::pair<int, int> minDistance, 
            std::pair<int, int> lowerBound, 
            std::pair<int, int> upperBound,        
            float maxPathSize = 30.0,
            float distToGoalTolerance = 5.0);

    void setPlanData(cudaPtr frame, Waypoint *goal, float velocity_m_s);

    bool search_init();
    bool loop();
    bool loop_optimize();
    bool goalReached();
    
    /// @brief Exports the current state of the graph as a vector
    /// @return vector, where each node = [x, z, node_type]
    std::vector<int3> exportGraphNodes();

    std::vector<Waypoint> getPlannedPath();

};

#endif