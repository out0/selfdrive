

#include "../include/fast_rrt.h"
#include <stdio.h>

extern "C"
{
    void *init(
        int width,
        int height,
        float og_real_width_m,
        float og_real_height_m,
        int min_dist_x,
        int min_dist_z,
        int lower_bound_ego_x,
        int lower_bound_ego_z,
        int upper_bound_ego_x,
        int upper_bound_ego_z,
        float max_steering_angle)
    {

        return new FastRRT(
            width,
            height,
            og_real_width_m,
            og_real_height_m,
            lower_bound_ego_x,
            lower_bound_ego_z,
            upper_bound_ego_x,
            upper_bound_ego_z,
            max_steering_angle);
    }

    void destroy(void *self)
    {
        FastRRT *f = (FastRRT *)self;
        delete f;
    }

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

        float3 start;
        start.x = start_x;
        start.y = start_z;
        start.z = start_heading;

        Memlist<float3> *curve = f->buildCurveWaypoints(start, velocity_m_s, sterr_angle, path_size);

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

    void connect_nodes_with_path_free(float *p) {
        delete []p;
    }

    float * connect_nodes_with_path(
        void *self,
        int start_x,
        int start_z,
        float start_heading,
        int end_x,
        int end_z,
        float end_heading,
        float velocity_m_s)
    {
        FastRRT *f = (FastRRT *)self;

        float3 start;
        start.x = start_x;
        start.y = start_z;
        start.z = start_heading;

        float3 end;
        end.x = end_x;
        end.y = end_z;
        end.z = end_heading;

        Memlist<float3> *curve = f->buildCurveWaypoints(start, end, velocity_m_s);
        int arr_size = 3 * curve->size + 1;

        float * res = new float[arr_size];

        for (int i = 0; i < curve->size; i++)
        {
            int pos = 3 * i + 1;
            res[pos] = curve->data[i].x;
            res[pos+1] = curve->data[i].y;
            res[pos+2] = curve->data[i].z;
        }
        
        res[0] = static_cast<float>(curve->size); 
        delete curve;
        return res;
    }
}
