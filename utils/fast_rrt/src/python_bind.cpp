#include "../include/fastrrt.h"

extern "C"
{

    void *fastrrt_initialize(int width, int height,
                             float perceptionWidthSize_m,
                             float perceptionHeightSize_m,
                             float maxSteeringAngle,
                             float vehicleLength,
                             int timeout_ms,
                             float maxPathSize = 30.0,
                             float distToGoalTolerance = 5.0)
    {
        return new FastRRT(width, height, perceptionWidthSize_m,
                           perceptionHeightSize_m, angle::rad(maxSteeringAngle),
                           vehicleLength, timeout_ms, maxPathSize, distToGoalTolerance);
    }

    void fastrrt_destroy(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        delete rrt;
    }

    bool is_planning(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        return rrt->isPlanning();
    }

    void set_plan_data(void *ptr, void *searchFramePtr, int goal_x, int goal_z, float heading_rad, float velocity_m_s)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        Waypoint p(goal_x, goal_z, angle::rad(heading_rad));
        rrt->setPlanData((SearchFrame *)searchFramePtr, &p, velocity_m_s);
    }

    void run(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        rrt->run();
    }

    void optimize(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        rrt->optimize();
    }

    void cancel(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        rrt->cancel();
    }

    bool goal_reached(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        return rrt->goalReached();
    }

    float *get_planned_path(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        std::vector<Waypoint> path = rrt->getPlannedPath();
        int size = path.size();

        float *res = new float[3 * size + 1];
        res[0] = (float)size;

        int i = 0;
        for (auto p : path)
        {
            int pos = (3 * i + 1);
            res[i] = p.x();
            res[i + 1] = p.z();
            res[i + 2] = p.heading().rad();
        }

        return res;
    }

    void release_planned_path_data(float *ptr)
    {
        delete[] ptr;
    }

    //  std::vector<Waypoint> getPlannedPath();
};