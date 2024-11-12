#ifndef H_FAST_RRT
#define H_FAST_RRT

#include <cuda_runtime.h>
#include "fast_rrt_mem.h"
#include <cuda_runtime.h>
#include "../src/kinematic_model.h"

class FastRRT
{
private:
    CurveGenerator *_curveGenerator;
    /* data */
public:
    Memlist<float3> *buildCurveWaypoints(float3 start, float velocity_meters_per_s, float steering_angle_deg, float path_size);
    Memlist<float3> *buildCurveWaypoints(float3 start, float3 end, float velocity_meters_per_s);
    FastRRT(
        int og_width, 
        int og_height, 
        float og_real_width_m, 
        float og_real_height_m,
        int lower_bound_x,
        int lower_bound_z,
        int upper_bound_x,
        int upper_bound_z,
        float max_steering_angle);
    ~FastRRT();
};



#endif