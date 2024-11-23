

#include "../include/fast_rrt.h"
#include "../src/cuda_frame.h"
#include <stdio.h>
#include <vector>

extern "C"
{
    void *init(
        int width,
        int height,
        float og_real_width_m,
        float og_real_height_m,
        int min_dist_x,
        int min_dist_z,
        int lower_bound_x,
        int lower_bound_z,
        int upper_bound_x,
        int upper_bound_z,
        float max_steering_angle_deg,
        int timeout_ms)
    {

        return new FastRRT(
            width,
            height,
            static_cast<double>(og_real_width_m),
            static_cast<double>(og_real_height_m),
            min_dist_x,
            min_dist_z,
            lower_bound_x,
            lower_bound_z,
            upper_bound_x,
            upper_bound_z,
            static_cast<double>(max_steering_angle_deg),
            timeout_ms);
    }

    void destroy(void *self)
    {
        FastRRT *f = (FastRRT *)self;
        delete f;
    }

    // RRT

    void set_plan_data(void *self, void *cuda_frame, int start_x, int start_z, float start_heading, int goal_x, int goal_z, float goal_heading, float velocity_m_s)
    {
        FastRRT *f = (FastRRT *)self;
        CudaFrame *cudaf = (CudaFrame *)cuda_frame;

        double3 start, goal;
        start.x = static_cast<double>(start_x);
        start.y = static_cast<double>(start_z);
        start.z = static_cast<double>(start_heading);
        goal.x = static_cast<double>(goal_x);
        goal.y = static_cast<double>(goal_z);
        goal.z = static_cast<double>(goal_heading);
        f->setPlanData(cudaf->getFramePtr(), start, goal, velocity_m_s);
    }
    void search(void *self) {
        FastRRT *f = (FastRRT *)self;
        f->search();
    }
    void cancel(void *self) {
        FastRRT *f = (FastRRT *)self;
        f->cancel();
    }

    bool is_planning(void *self) {
        FastRRT *f = (FastRRT *)self;
        return f->isPlanning();
    }

    
    int get_path_size(void *self) {
        FastRRT *f = (FastRRT *)self;
        return f->getPathSize();
    }

    void get_path(void *self, float *result) {
        FastRRT *f = (FastRRT *)self;
        f->copyPathTo(result);
    }

    //
    // TESTING STUFF
    //

    int gen_path_waypoint(
        void *self,
        float *res,
        int start_x,
        int start_z,
        float start_heading,
        float velocity_m_s,
        float sterr_angle,
        float path_size)
    {
        FastRRT *f = (FastRRT *)self;

        double3 start;
        start.x = static_cast<double>(start_x);
        start.y = static_cast<double>(start_z);
        start.z = static_cast<double>(start_heading);

        std::vector<double3> curve = f->buildCurveWaypoints(
            start, 
            static_cast<double>(velocity_m_s), 
            static_cast<double>(sterr_angle), 
            static_cast<double>(path_size));

        int pos = 0;
        for (double3 p : curve)
        {
            res[pos] = static_cast<float>(p.x);
            res[pos + 1] = static_cast<float>(p.y);
            res[pos + 2] = static_cast<float>(p.z);
            pos += 3;
        }
        return curve.size();
    }

    void connect_nodes_with_path_free(double *p)
    {
        delete[] p;
    }

    float *connect_nodes_with_path(
        void *self,
        int start_x,
        int start_z,
        float start_heading,
        int end_x,
        int end_z,
        float velocity_m_s)
    {
        FastRRT *f = (FastRRT *)self;

        double3 start;
        start.x = static_cast<double>(start_x);
        start.y = static_cast<double>(start_z);
        start.z = static_cast<double>(start_heading);

        double3 end;
        end.x = static_cast<double>(end_x);
        end.y = static_cast<double>(end_z);
        end.z = 0.0;

        std::vector<double3> curve = f->buildCurveWaypoints(
            start,
            end, 
            static_cast<double>(velocity_m_s));

        float *res = new float[3 * curve.size() + 1];

        int pos = 1;
        for (double3 p : curve)
        {
            res[pos] = static_cast<float>(p.x);
            res[pos + 1] = static_cast<float>(p.y);
            res[pos + 2] = static_cast<float>(p.z);
            pos += 3;
        }

        res[0] = static_cast<float>(curve.size());
        return res;
    }

    void test_draw_path(void *self, void *cuda_frame, int x1, int z1, double h1, int x2, int z2, double h2)
    {
        FastRRT *f = (FastRRT *)self;
        CudaFrame *cudaf = (CudaFrame *)cuda_frame;
        double3 start, end;
        start.x = x1;
        start.y = z1;
        start.z = h1;
        end.x = x2;
        end.y = z2;
        end.z = h2;
        f->testDrawPath(cudaf->getFramePtr(), start, end);
    }

}
