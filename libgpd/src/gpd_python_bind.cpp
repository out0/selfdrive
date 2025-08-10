#include "../include/gpd.h"

extern "C"
{

    float * find_lowest_cost_waypoint_for_heading(void *searchFrame, int minDistX, int minDistZ, int goalX, int goalZ, float heading_rad, bool compute_exclusion_zone)
    {
        GoalPointDiscover gpd;
        SearchFrame *f = (SearchFrame *)searchFrame;
        if (compute_exclusion_zone)
            gpd.computeExclusionZone(*f, angle::rad(heading_rad));

        Waypoint p = gpd.findLowestCostWaypointWithHeading(*f, {minDistX, minDistZ}, goalX, goalZ, heading_rad);

        float * res = new float[3];
        res[0] = p.x();
        res[1] = p.z();
        res[2] = p.heading().rad();

        return res;
    }

    float * find_lowest_cost_waypoint_direct_to_goal(void *searchFrame, int minDistX, int minDistZ, int goalX, int goalZ, float next_heading, bool compute_exclusion_zone)
    {
        GoalPointDiscover gpd;
        SearchFrame *f = (SearchFrame *)searchFrame;
        if (compute_exclusion_zone)
            gpd.computeExclusionZone(*f, angle::rad(next_heading));

        Waypoint p = gpd.findLowestCostWaypointToGoal(*f, {minDistX, minDistZ}, goalX, goalZ, next_heading);

        float * res = new float[3];
        res[0] = p.x();
        res[1] = p.z();
        res[2] = p.heading().rad();

        return res;
    }

    float * find_lowest_error_waypoint_to_goal(void *searchFrame, int minDistX, int minDistZ, int goalX, int goalZ, float heading_rad, bool compute_exclusion_zone)
    {
        GoalPointDiscover gpd;
        SearchFrame *f = (SearchFrame *)searchFrame;
        if (compute_exclusion_zone)
            gpd.computeExclusionZone(*f, angle::rad(heading_rad));
            
        Waypoint p = gpd.findLowestErrorWaypointToGoal(*f, {minDistX, minDistZ}, goalX, goalZ, heading_rad);
        float * res = new float[3];
        res[0] = p.x();
        res[1] = p.z();
        res[2] = p.heading().rad();

        return res;
    }


    void free_waypoint_res(float *ptr) {
        delete []ptr;
    }
}