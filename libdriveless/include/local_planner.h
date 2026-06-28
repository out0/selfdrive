#ifndef H__LOCAL_PLANNER_
#define H__LOCAL_PLANNER_

#include <tuple>
#include <vector>
#include "waypoint.h"

class LocalPlanner {
    /// @brief Initializes the local planner
    /// @param copyIntrinsicCostsFromFrame copys the values in frame's channel G as intrinsic values to support using cost maps.
    virtual void initialize(bool copyIntrinsicCostsFromFrame = false) = 0;
    
    /// @brief Executes a planning loop
    /// @return false if the planner should stop planning
    virtual bool planning_loop() = 0;

    /// @brief Executes a optimization loop
    /// @return false if the planner should stop optimizing
    virtual bool path_optimize_loop() = 0;

    /// @brief Checks if the planner reached the goal
    /// @return true in case of goal reached
    virtual bool goalReached() = 0;

    /// @brief Returns the planned path and the path cost
    /// @return a tuple with a vector of waypoints and a float representing the total path cost
    virtual std::tuple<std::vector<Waypoint>, float> getPlannedPath() = 0;

    /// @brief Returns an interpolation of the planned path and the original path cost
    /// @return  a tuple with a vector of waypoints and a float representing the total path cost
    virtual std::tuple<std::vector<Waypoint>, float> getInterpolatedPlannedPath() = 0;
};


#endif