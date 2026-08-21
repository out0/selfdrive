/*
Curve generation
*/
#include <driveless/cuda_basic.h>
#include <driveless/frame_params.h>
#include <driveless/math_utils.h>

__device__ __host__ float distance(float3 p1, float3 p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    return sqrtf(dx*dx + dy*dy);
}


typedef float (*interpolation_callback)(void *, int, int, float);


/// @brief Interpolates an hermite curve
/// @param plane_dim plane dimensions (width, height)
/// @param p1 (x, y, heading)
/// @param p2 (x, y, heading)
/// @param wheelbase distance between the two wheels
/// @param delta_max_rad max wheel-turning angle in radians
/// @param cb callback function for each x,y,heading interpolated point
/// @param result_ptr callback result pointer for permanent results
/// @return 
__device__ __host__ float hermite_curve(int2 plane_dim, float3 p1, float3 p2,
                                        float wheelbase, float delta_max_rad, interpolation_callback cb, void * result_ptr)
{

    const int plane_width = plane_dim.x;
    const int plane_height = plane_dim.y;

    float d = distance(p1, p2);
    float kappa_max = tanf(delta_max_rad) / wheelbase;

    float a1 = p1.z - PI / 2;
    float a2 = p2.z - PI / 2;

    float2 tan1 = {d * cosf(a1), d * sinf(a1)};
    float2 tan2 = {d * cosf(a2), d * sinf(a2)};

    int maxPoints = 2 * TO_INT(d);
    if (maxPoints < 2)
        return -1;

    int last_x = -1;
    int last_z = -1;

    float curve_cost = 0;

    for (int i = 0; i < maxPoints; ++i)
    {
        float t = TO_FLOAT(i) / (maxPoints - 1);
        float t2 = t * t;
        float t3 = t2 * t;

        // Position basis
        float h00 = 2*t3 - 3*t2 + 1;
        float h10 = t3 - 2*t2 + t;
        float h01 = -2*t3 + 3*t2;
        float h11 = t3 - t2;

        float x = h00*p1.x + h10*tan1.x + h01*p2.x + h11*tan2.x;
        float z = h00*p1.y + h10*tan1.y + h01*p2.y + h11*tan2.y;

        // First derivative basis
        float h00d = 6*t2 - 6*t;
        float h10d = 3*t2 - 4*t + 1;
        float h01d = -6*t2 + 6*t;
        float h11d = 3*t2 - 2*t;

        float xp = h00d*p1.x + h10d*tan1.x + h01d*p2.x + h11d*tan2.x;
        float zp = h00d*p1.y + h10d*tan1.y + h01d*p2.y + h11d*tan2.y;

        // Second derivative basis
        float h00dd = 12*t - 6;
        float h10dd = 6*t - 4;
        float h01dd = -12*t + 6;
        float h11dd = 6*t - 2;

        float xpp = h00dd*p1.x + h10dd*tan1.x + h01dd*p2.x + h11dd*tan2.x;
        float zpp = h00dd*p1.y + h10dd*tan1.y + h01dd*p2.y + h11dd*tan2.y;

        // Curvature check — bail immediately, curve gets discarded by caller
        float denom = powf(xp*xp + zp*zp, 1.5f);
        float kappa = (denom > 1e-6f) ? fabsf(xp*zpp - zp*xpp) / denom : 0.0f;
        if (kappa > kappa_max)
            return -1;

        if (x < 0 || x >= plane_width || z < 0 || z >= plane_height)
            continue;

        int cx = TO_INT(x);
        int cz = TO_INT(z);
        if (cx == last_x && cz == last_z) continue;
        if (cx < 0 || cx >= plane_width || cz < 0 || cz >= plane_height) continue;

        float heading = atan2f(zp, xp) + HALF_PI;

        float point_cost = cb(result_ptr, cx, cz, heading);

        if (point_cost < 0)
            return -1;

        last_x = cx;
        last_z = cz;
        curve_cost += point_cost;
    }

    return curve_cost;
}

void kinematic_curve(int2 start, float heading, int *params, float *physical_params, interpolation_callback cb, void * result_ptr)
{
    float wheelbase = physical_params[PHYSICAL_PARAM_WHEELBASE_PX];
    float delta_max_rad = physical_params[PHYSICAL_PARAM_MAX_STEERING_RAD];

    float3 p1 = {TO_FLOAT(start.x), TO_FLOAT(start.y), heading};
    float3 p2 = {TO_FLOAT(params[0]), TO_FLOAT(params[1]), TO_FLOAT(params[2])};

    hermite_curve(make_int2(params[3], params[4]), p1, p2, wheelbase, delta_max_rad, cb, result_ptr);

}