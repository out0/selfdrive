#pragma once

#ifndef __SEARCH_FRAME_DRIVELESS_H
#define __SEARCH_FRAME_DRIVELESS_H

#include "cuda_frame.h"
#include "cuda_params.h"
#include "waypoint.h"
#include "moving_obstacle.h"
#include "cuda_ptr.h"

// CODE:BEGIN

#include <vector>
#include <memory>
typedef unsigned char uchar;

#define PATH_FEASIBLE_CPU_THRESHOLD 20

class SearchFrame : public CudaFrame<float3>
{
private:
    cptr<int> _params;
    cptr<uchar3> _classColors;
    cptr<float> _classCosts;
    cptr<int> _bestValue;
    int _classCount;
    bool _safeZoneChecked;
    bool _safeZoneVectorialChecked;
    std::pair<int, int> _minDistanceChecked;

    std::pair<int, int> checkTraversableAngleBitPairCheck(float heading_rad);

    // std::vector<bool> checkFeasiblePathCPU(std::vector<Waypoint> path, bool computeHeadings);
    // std::vector<bool> checkFeasiblePathGPU(std::vector<Waypoint> path, bool computeHeadings);

public:
    SearchFrame(int width, int height, std::pair<int, int> lowerBound, std::pair<int, int> upperBound);
    ~SearchFrame();

    void clear() override;

    void copyFrom(float *ptr) override;
    void copyTo(float *ptr);

    inline bool isSafeZoneChecked()
    {
        return _safeZoneChecked;
    }
    // inline std::pair<int, int> minDistance()
    // {
    //     return {_params[FRAME_PARAM_MIN_DIST_X], _params[FRAME_PARAM_MIN_DIST_Z]};
    // }

    inline std::pair<int, int> lowerBound()
    {
        return {_params->get()[FRAME_PARAM_LOWER_BOUND_X], _params->get()[FRAME_PARAM_LOWER_BOUND_Z]};
    }

    inline std::pair<int, int> upperBound()
    {
        return {_params->get()[FRAME_PARAM_UPPER_BOUND_X], _params->get()[FRAME_PARAM_UPPER_BOUND_Z]};
    }

    /// @brief Sets the colors for each class, to allow conversion using exportToColorFrame(). Each class is an int starting at 0. The colors are sequential: the first entry correspond to the color for class 0, the second entry for class 1... so one, so forth.
    /// @param classColors
    void setClassColors(std::vector<std::tuple<int, int, int>> classColors);

    /// @brief Exports the current frame to dest as a colored frame, based on the segmentation class in frame[i].x and on the conversion colors given by setClassColors()
    /// @param dest
    /// @return true if the export could be performed. The export can fail if we cannot allocate enough GPU memory to perform the conversion task
    bool exportToColorFrame(uchar *dest);

    /// @brief Sets the cost for each class. Negative cost means infinite cost (obstacle); the costs are sequential: first entry is for for class 0, second entry is for class 1, etc.
    /// @param classCosts
    void setClassCosts(std::vector<float> classCosts);

    /// @brief Returns the cost associated with a segmentation class by setClassCosts()
    /// @param segClass
    /// @return
    float getClassCost(unsigned int classId);

    /// @brief Returns the cost associated with a segmentation class by setClassCosts()
    /// @param x
    /// @param z
    /// @return
    float getClassCostForNode(int x, int z);
    /// @brief Returns the cost associated with a segmentation class by setClassCosts()
    /// @param pos 
    /// @return 
    float getClassCostForNode(long pos);

    /// @brief check if the pos x,z is an obstacle (class only, it does not check if it is travessable)
    /// @param x
    /// @param z
    /// @return
    bool isObstacle(int x, int z);

    /// @brief checks if the pos x,z is travesable (for any angle)
    /// @param x
    /// @param z
    /// @return
    bool isTraversable(int x, int z);

    /// @brief checks if the pos x,z is travesable in angle a
    /// @param x
    /// @param z
    /// @return
    bool isTraversable(int x, int z, angle a, bool precision_check);

    /// @brief Returns the pre-computed (using setGoal()) cost for pos x,z, given by its distance to the goal, multiplied by the class cost provided by function setClassCosts()
    /// @param x
    /// @param z
    /// @return
    double getCost(int x, int z);

    /// @brief Sets a goal point to pre-compute distance cost and traversability for every x,z position in the search space.
    /// @param x
    /// @param z
    // void setGoal(std::pair<int, int> minDistance, int x, int z, bool computeVectorized);

    /// @brief Sets a goal point to pre-compute distance cost and traversability for every x,z position in the search space.
    /// @param x
    /// @param z
    void processSafeDistanceZone(std::pair<int, int> minDistance, bool computeVectorized);

    /// @brief Returns true if a path is feasible. Otherwise, returns false
    /// @param path
    /// @param computeHeadings should compute headings when calculating feasibility
    /// @return
    bool checkFeasiblePath(float *points, int count, int minDistX, int minDistZ, bool informWaypointIndividualFeasibility = false);

    /// @brief Returns true if a path is feasible. Otherwise, returns false
    /// @param path
    /// @param minDistX 
    /// @param minDistZ 
    /// @return 
    bool checkFeasiblePath(std::vector<Waypoint> &path, int minDistX, int minDistZ, bool informWaypointIndividualFeasibility = false);


    /// @brief Sets moving obstacles
    /// @param obstacles
    void setMovingObstacles(std::vector<MovingObstacle> obstacles);

    // /// @brief Finds the closest waypoint, with heading as close to zero as possible
    // /// @param goal_x
    // /// @param goal_z
    // /// @return
    // Waypoint findBestWaypoint(int goal_x, int goal_z);

    // /// @brief Finds the closest waypoint that is feasible for the given heading
    // /// @param goal_x
    // /// @param goal_z
    // /// @param heading
    // /// @return
    // Waypoint findBestWaypoint(int goal_x, int goal_z, float heading);

    /// @brief Returns the params array that are used to setup this search frame execution
    /// @return
    inline int *getCudaFrameParamsPtr()
    {
        return _params->get();
    }

    /// @brief Returns the class cost array that hold the cost of each class of segmentation
    /// @return
    inline float *getCudaClassCostsPtr()
    {
        return _classCosts->get();
    }


    // returns the vehicle length in meters (computed from the values set by lower and upper bound and the height perception size in meters (to properly convert px to m))
    double computeVehicleLength(double perceptionHeightSize_m);

    /// @brief Sets a new value to a x,z position of the search frame
    /// @param x x position
    /// @param z z position
    /// @param v1 first channel value
    /// @param v2 second channel value
    /// @param v3 third channel value
    void setValues(int x, int z, float v1, float v2, float v3);

    static bool computePathHeadings(int width, int height, std::vector<Waypoint> &waypoints);

    /// @brief Computes the distance between every pixel in the search frame and an arbitrary (x, z) position
    /// @param x 
    /// @param z 
    void processDistanceToGoal(int x, int z);

    /// @brief Returns the computed distance to (x, z) goal (processDistanceToGoal it is required first, to pre-compute the distances)
    /// @param x 
    /// @param z 
    /// @return 
    float getDistanceToGoal(int x, int z);
};

// CODE:END

#endif