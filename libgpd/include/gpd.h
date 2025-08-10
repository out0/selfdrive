#pragma once

#ifndef __GOAL_POINT_DISCOVER_DRIVELESS_H
#define __GOAL_POINT_DISCOVER_DRIVELESS_H

#include <driveless/waypoint.h>
#include <driveless/cuda_basic.h>
#include <driveless/search_frame.h>

class GoalPointDiscover 
{
private:
    cptr<int> _bestValue;
    bool _computeExclusionZone;
    

public:
    /// @brief 
    /// @param computeExclusionZone experimental: exclusion zone is used to redirect the goal planning to avoid planning behind obstacles
    GoalPointDiscover(bool computeExclusionZone = false);

    void computeExclusionZone(SearchFrame &frame, angle heading);

    /// @brief Finds the closest waypoint, with heading as close to zero as possible
    /// @param goal_x
    /// @param goal_z
    /// @return
    Waypoint findLowestCostWaypointToGoal(SearchFrame &frame, std::pair<int, int> minDist, int goal_x, int goal_z, float next_heading);

    /// @brief Finds the least error waypoint to goal, in terms of heading and distance
    /// @param goal_x
    /// @param goal_z
    /// @return
    Waypoint findLowestErrorWaypointToGoal(SearchFrame &frame, std::pair<int, int> minDist, int goal_x, int goal_z, float best_heading);


    /// @brief Finds the closest waypoint that is feasible for the given heading
    /// @param goal_x
    /// @param goal_z
    /// @param heading
    /// @return
    Waypoint findLowestCostWaypointWithHeading(SearchFrame &frame, std::pair<int, int> minDist, int goal_x, int goal_z, float heading);


};

// CODE:END

#endif