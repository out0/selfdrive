#ifndef H_KINEMATIC_MODEL
#define H_KINEMATIC_MODEL

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

class CurveGenerator
{

public:
    static std::vector<double3> buildCurveWaypoints(
        double3 _center,
        double _rate_w,
        double _rate_h,
        double _lr,
        double _max_steering_angle_deg,
        double3 firstPos, 
        double velocity_meters_per_s, 
        double steering_angle_deg, 
        double path_size);

    static std::vector<double3> buildCurveWaypoints(
        double3 _center,
        double _rate_w,
        double _rate_h,
        double _lr,
        double _max_steering_angle_deg,       
        double3 start, 
        double3 end, 
        double velocity_meters_per_s);
};

#endif