#pragma once

#include "cuda_basic.h"



class CudaFrame
{
    float3 *frame;
    int width;
    int height;
    int min_dist_x;
    int min_dist_z;
    int lower_bound_x;
    int lower_bound_z;
    int upper_bound_x;
    int upper_bound_z;
    void copyToCpuPointer(float3 *source, float *target);
    void copyToCpuPointer(uchar3 *source, u_char *target);
    void checkFeasibleWaypointsCPU(float *points, int count, bool computeHeadings);
    //void computeHeadings(float *points, int count);

public:
    CudaFrame(float *ptr, int width, int height, int min_dist_x, int min_dist_z, int lower_bound_x, int lower_bound_z, int upper_bound_x, int upper_bound_z);
    ~CudaFrame();

    int getWidth();
    int getHeight();

    void convertToColorFrame(u_char *dest);
    void setGoal(int goal_x, int goal_z);
    void copyBack(float *img);
    void setGoalVectorized(int goal_x, int goal_z);

    /// @brief Check which waypoints are feasible
    /// @param points a four-position float array, where the first 3 channels represent x,z,heading pose and the last channel is to be rewritten carrying 1 for feasible or 0 for not-feasible
    /// @param count how many points are in the array (total float * should be count * sizeof(float) * 4)
    void checkFeasibleWaypoints(float *points, int count, bool computeHeadings);

    static int get_class_cost(int segmentation_class);

    float * bestWaypointPosForHeading(int goal_x, int goal_z, float heading);

    float * bestWaypointPos(int goal_x, int goal_z);

    float *bestWaypointInDirection(int start_x, int start_z, int goal_x, int goal_z);

    bool checkWaypointClassIsObstacle(int x, int z);

    float getCost(int x, int z);

    float3 * getFramePtr();
};