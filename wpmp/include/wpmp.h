#pragma once

#ifndef __WPMP_DRIVELESS_H
#define __WPMP_DRIVELESS_H

#include <cmath>
#include <chrono>
#include <driveless/angle.h>
#include <driveless/waypoint.h>
#include <driveless/frame.h>
#include <driveless/search_params.h>
#include <driveless/local_planner.h>
#include <vector>
#include <tuple>

class WPMP : public LocalPlanner
{
private:
    

public:
    WPMP(EgoParams &egoParams, bool smartExpansion = true);

    void setPlanData(SearchParams &params);

    /// @brief Initializes the local planner
    /// @param copyIntrinsicCostsFromFrame copys the values in frame's channel G as intrinsic values to support using cost maps.
    virtual void initialize(bool copyIntrinsicCostsFromFrame = false) override;

    /// @brief Executes a planning loop
    /// @return false if the planner should stop planning
    bool planning_loop() override;

    /// @brief Executes a optimization loop
    /// @return false if the planner should stop optimizing
    bool path_optimize_loop() override;

    /// @brief Checks if the planner reached the goal
    /// @return true in case of goal reached
    bool goalReached() override;

    /// @brief Exports the current state of the graph as a vector
    /// @return vector, where each node = [x, z, node_type]
    std::vector<GraphNode> exportGraphNodes();

    /// @brief Returns the heading of a node in the internal graph
    /// @param x x-coordinate of the node
    /// @param z z-coordinate of the node
    /// @return the heading angle stored for that node
    angle getHeading(int x, int z);

    /// @brief Returns the planned path and the path cost
    /// @return a tuple with a vector of waypoints and a float representing the total path cost
    std::tuple<std::vector<Waypoint>, float> getPlannedPath() override;

    /// @brief Returns an interpolation of the planned path and the original path cost
    /// @return  a tuple with a vector of waypoints and a float representing the total path cost
    std::tuple<std::vector<Waypoint>, float> getInterpolatedPlannedPath();

    std::vector<Waypoint> interpolatePlannedPath(std::vector<Waypoint> path);
    std::vector<Waypoint> idealGeometryCurveNoObstacles(Waypoint goal);
    void computeGraphRegionDensity();

    /// @brief Saves the planner current Graph state, to be used when debugging the algorithm execution
    /// @param filename
    void saveCurrentGraphState(std::string filename);

    /// @brief Loads planner current Graph state from file, to be used when debugging the algorithm execution
    /// @param filename
    void loadGraphState(std::string filename);
};

#endif