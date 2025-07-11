#include "../include/fastrrt.h"
#include "../../cudac/include/cuda_frame.h"

extern "C"
{
    void *fastrrt_initialize(
        int width,
        int height,
        float perceptionWidthSize_m,
        float perceptionHeightSize_m,
        float maxSteeringAngle_deg,
        float vehicleLength,
        int timeout_ms,
        int minDistance_x, int minDistance_z,
        int lowerBound_x, int lowerBound_z,
        int upperBound_x, int upperBound_z,
        float maxPathSize,
        float distToGoalTolerance)
    {
        return new FastRRT(
            width, height,
            perceptionWidthSize_m,
            perceptionHeightSize_m,
            angle::deg(maxSteeringAngle_deg),
            vehicleLength,
            timeout_ms,
            {minDistance_x, minDistance_z},
            {lowerBound_x, lowerBound_z},
            {upperBound_x, upperBound_z},
            maxPathSize,
            distToGoalTolerance);
    }

    void fastrrt_destroy(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        delete rrt;
    }

    void set_plan_data(void *ptr, void *cudaFramePtr, int start_x, int start_z, float start_heading_rad, int goal_x, int goal_z, float goal_heading_rad, float velocity_m_s)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        Waypoint s(start_x, start_z, angle::rad(start_heading_rad));
        Waypoint p(goal_x, goal_z, angle::rad(goal_heading_rad));
        // printf ("p.x = %d, p.y = %d, p.h = %f\n", p.x(), p.z(), p.heading().deg());

        CudaFrame *frame = (CudaFrame *)cudaFramePtr;
        rrt->setPlanData(frame->getFramePtr(), s, p, velocity_m_s);
    }

    bool goal_reached(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        return rrt->goalReached();
    }

    void search_init(void *ptr, bool copyIntrinsicCostsFromFrame)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        rrt->search_init(copyIntrinsicCostsFromFrame);
    }
    bool loop(void *ptr, bool smartExpansion)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        return rrt->loop(smartExpansion);
    }

    bool loop_optimize(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        return rrt->loop_optimize();
    }

    int *export_graph_nodes(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        auto nodes = rrt->exportGraphNodes();

        int *res = new int[3 * nodes.size() + 1];

        res[0] = nodes.size();

        int i = 1;
        for (auto n : nodes)
        {
            res[i] = n.x;
            res[i + 1] = n.y;
            res[i + 2] = n.z;
            i += 3;
        }

        return res;
    }
    void release_export_graph_nodes(float *ptr)
    {
        delete[] ptr;
    }

    float *convertPath(std::vector<Waypoint> &path)
    {
        int size = path.size();

        // printf("size = %d\n", size);

        float *res = new float[3 * size + 1];
        res[0] = (float)size;

        // printf("res[0] = %f\n", res[0]);

        int i = 0;
        for (auto p : path)
        {
            int pos = (3 * i + 1);
            res[pos] = p.x();
            res[pos + 1] = p.z();
            res[pos + 2] = p.heading().rad();
            i += 1;
        }

        return res;
    }

    float *get_planned_path(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        std::vector<Waypoint> path = rrt->getPlannedPath();
        return convertPath(path);
    }

    float *interpolate_planned_path(void *ptr)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        std::vector<Waypoint> path = rrt->interpolatePlannedPath();
        return convertPath(path);
    }

    void release_planned_path_data(float *ptr)
    {
        delete[] ptr;
    }

    float *interpolate_planned_path_p(void *ptr, float *p, int size)
    {
        FastRRT *rrt = (FastRRT *)ptr;

        std::vector<Waypoint> pref;

        for (int i = 0; i < size; i += 3)
        {
            pref.push_back(Waypoint(p[i], p[i + 1], angle::rad(p[i + 2])));
            printf("(%d, %d, %f)\n", (int)p[i], (int)p[i + 1], p[i + 2]);
        }

        std::vector<Waypoint> path = rrt->interpolatePlannedPath(pref);
        return convertPath(path);
    }

    float *ideal_curve(void *ptr, int goal_x, int goal_z, float goal_heading_rad)
    {
        FastRRT *rrt = (FastRRT *)ptr;
        std::vector<Waypoint> path = rrt->idealGeometryCurveNoObstacles({goal_x,
                                                                         goal_z,
                                                                         angle::rad(goal_heading_rad)});
        return convertPath(path);
    }


    void compute_region_debug_performance(void *ptr) {
        FastRRT *rrt = (FastRRT *)ptr;
        rrt->__computeGraphRegionDensity();
    }
};