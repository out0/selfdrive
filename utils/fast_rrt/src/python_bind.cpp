

#include "../include/fast_rrt.h"
#include "../src/cuda_frame.h"
#include <stdio.h>

extern "C"
{
    void *init(
        int width,
        int height,
        double og_real_width_m,
        double og_real_height_m,
        int min_dist_x,
        int min_dist_z,
        int lower_bound_ego_x,
        int lower_bound_ego_z,
        int upper_bound_ego_x,
        int upper_bound_ego_z,
        double max_steering_angle,
        double velocity_m_s)
    {

        return new FastRRT(
            width,
            height,
            og_real_width_m,
            og_real_height_m,
            min_dist_x,
            min_dist_z,
            lower_bound_ego_x,
            lower_bound_ego_z,
            upper_bound_ego_x,
            upper_bound_ego_z,
            max_steering_angle,
            velocity_m_s);
    }

    void destroy(void *self)
    {
        FastRRT *f = (FastRRT *)self;
        delete f;
    }

    int gen_path_waypoint(
        void *self,
        double *res,
        int start_x,
        int start_z,
        double start_heading,
        double velocity_m_s,
        double sterr_angle,
        double path_size)
    {
        FastRRT *f = (FastRRT *)self;

        double3 start;
        start.x = start_x;
        start.y = start_z;
        start.z = start_heading;

        Memlist<double3> *curve = f->buildCurveWaypoints(start, velocity_m_s, sterr_angle, path_size);

        for (int i = 0; i < curve->size; i++)
        {
            int pos = 3 * i;
            res[pos] = curve->data[i].x;
            res[pos+1] = curve->data[i].y;
            res[pos+2] = curve->data[i].z;
        }
        
        int s = curve->size;
        delete curve;
        return s;
    }

    void connect_nodes_with_path_free(double *p) {
        delete []p;
    }

    double * connect_nodes_with_path(
        void *self,
        int start_x,
        int start_z,
        double start_heading,
        int end_x,
        int end_z,
        double end_heading,
        double velocity_m_s)
    {
        FastRRT *f = (FastRRT *)self;

        double3 start;
        start.x = start_x;
        start.y = start_z;
        start.z = start_heading;

        double3 end;
        end.x = end_x;
        end.y = end_z;
        end.z = end_heading;

        Memlist<double3> *curve = f->buildCurveWaypoints(start, end, velocity_m_s);
        int arr_size = 3 * curve->size + 1;

        double * res = new double[arr_size];

        for (int i = 0; i < curve->size; i++)
        {
            int pos = 3 * i + 1;
            res[pos] = curve->data[i].x;
            res[pos+1] = curve->data[i].y;
            res[pos+2] = curve->data[i].z;
        }
        
        res[0] = static_cast<double>(curve->size); 
        delete curve;
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
