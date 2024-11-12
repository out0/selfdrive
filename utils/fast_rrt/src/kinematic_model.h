#ifndef H_KINEMATIC_MODEL
#define H_KINEMATIC_MODEL

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include "../include/fast_rrt_mem.h"

class CurveGenerator {
    float3 _center;
    float _rate_w; 
    float _rate_h;
    float _lr;
    float _max_steering_angle_deg;
    


public:
    CurveGenerator(float3 center, float rate_w, float rate_h, float lr, float max_steering_angle_deg);
    Memlist<float3>* buildCurveWaypoints(float3 firstPos, float velocity_meters_per_s, float steering_angle_deg, float path_size);
    Memlist<float3>* buildCurveWaypoints(float3 start, float3 end, float velocity_meters_per_s);


};


#endif