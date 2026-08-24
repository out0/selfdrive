#include <driveless/waypoint.h>
#include <vector>
#include <driveless/cuda_basic.h>

typedef float (*interpolation_callback)(void *, int, int, float);

extern float hermite_curve(int2 plane_dim, float3 p1, float3 p2,
                           float wheelbase, float delta_max_rad,
                           interpolation_callback cb, void *result_ptr);

extern float kinematic_curve(int2 start, float heading, float steering_angle,
                             float velocity_m_s, float max_path_size, int *params,
                             float *physical_params, interpolation_callback cb, void *result_ptr);

static float collect_waypoint(void *ctx, int x, int z, float heading)
{
    std::vector<Waypoint> *out = static_cast<std::vector<Waypoint> *>(ctx);
    out->push_back({x, z, angle::rad(heading)});
    return 1;
};

/// @brief Interpolates an hermite curve
/// @param plane_width plane width
/// @param plane_height plane height
/// @param p1 (x, y, heading)
/// @param p2 (x, y, heading)
/// @param wheelbase distance between the two wheels
/// @param delta_max_rad max wheel-turning angle in radians
/// @return
std::vector<Waypoint> hermite_interpolatior(int plane_width,
                                            int plane_height,
                                            Waypoint p1,
                                            Waypoint p2,
                                            float wheelbase,
                                            float delta_max_rad)
{
    std::vector<Waypoint> res;

    float3 fp1 = {static_cast<float>(p1.x()), static_cast<float>(p1.z()), static_cast<float>(p1.heading().rad())};
    float3 fp2 = {static_cast<float>(p2.x()), static_cast<float>(p2.z()), static_cast<float>(p2.heading().rad())};

    float cost = hermite_curve({plane_width, plane_height}, fp1, fp2,
                               wheelbase, delta_max_rad,
                               &collect_waypoint, &res);

    if (cost < 0)
        res.clear(); // curve exceeded steering limit — discard partial result

    return res;
}

extern "C"
{

    // Flat, ctypes-callable wrapper.
    // p1/p2 are passed as separate (x, z, heading_rad) floats instead of a
    // float3 struct to avoid any struct-layout/ABI ambiguity from Python.
    //
    // Returns true if the curve stayed within the steering limit (result is
    // valid / should be kept), false otherwise (caller should discard whatever
    // the callback collected, mirroring the `res.clear()` behavior in the
    // original C++ wrapper).
    bool hermite_interpolate_c(int plane_width, int plane_height,
                               float p1_x, float p1_z, float p1_heading_rad,
                               float p2_x, float p2_z, float p2_heading_rad,
                               float wheelbase, float delta_max_rad,
                               interpolation_callback cb, void *result_ptr)
    {
        float3 fp1 = {p1_x, p1_z, p1_heading_rad};
        float3 fp2 = {p2_x, p2_z, p2_heading_rad};

        float cost = hermite_curve({plane_width, plane_height}, fp1, fp2,
                                   wheelbase, delta_max_rad, cb, result_ptr);

        return cost >= 0;
    }

    bool kinematic_interpolate_c(int plane_width, int plane_height,
                                 float p1_x, float p1_z, float p1_heading_rad,
                                 float steering_angle, float velocity_m_s,
                                 float max_path_size, int *params,
                                 float *physical_params,
                                 interpolation_callback cb, void *result_ptr)
    {

        float3 fp1 = {p1_x, p1_z, p1_heading_rad};

        return kinematic_curve({TO_INT(fp1.x), TO_INT(fp1.y)}, p1_heading_rad,
                               steering_angle, velocity_m_s, max_path_size, params, physical_params, cb, result_ptr);
    }
}
