#include <driveless/cuda_basic.h>
#include <driveless/search_zone_utils.h>

__device__ __host__ int2 zone_location(int2 zone_dim_size, int2 zone_grid_size, int x, int z)
{
    int xg = x / zone_dim_size.x;
    int zg = z / zone_dim_size.y;

    if (xg >= zone_grid_size.x)
        xg = zone_grid_size.x - 1;
    if (zg >= zone_grid_size.y)
        zg = zone_grid_size.y - 1;

    return {xg, zg};
}

__device__ __host__ bool is_zone_border(int x, int z, int xg, int zg, int search_zone_dim_w, int search_zone_dim_h)
{
    const int A = xg * search_zone_dim_w;
    const int B = zg * search_zone_dim_h;

    return (x >= A && x <= (A + search_zone_dim_w - 1)) && (z == B || z == (B + search_zone_dim_h - 1)) ||
           (z >= B && z <= (B + search_zone_dim_h - 1)) && (x == A || x == (A + search_zone_dim_w - 1));
}

__device__ __host__ bool is_zone_edge(int x, int z, int xg, int zg, int2 search_zone_dim)
{
    const int A = xg * search_zone_dim.x;
    const int B = zg * search_zone_dim.y;

    return (x == A || x == (A + search_zone_dim.x - 1)) && (z == B || z == (B + search_zone_dim.y - 1));
}

__device__ __host__ int4 sz_egdes_frame_pos(int2 sz_location, int2 search_zone_dim, int frame_width)
{
    const int A = sz_location.x * search_zone_dim.x;
    const int B = sz_location.y * search_zone_dim.y;
    const int tl = COMPUTE_POS(frame_width, A, B);
    const int tr = COMPUTE_POS(frame_width, (A + search_zone_dim.x - 1), B);
    const int bl = COMPUTE_POS(frame_width, A, (B + search_zone_dim.y - 1));
    const int br = COMPUTE_POS(frame_width, (A + search_zone_dim.x - 1), (B + search_zone_dim.y - 1));
    return {tl, tr, bl, br};
}

#define DELTA_T 0.1

__device__ __host__ bool is_reachable(int *params, double *physical_params, int2 goal, float goal_heading, bool positive_steer, float velocity_px_s, int x, int z)
{
    const double lr = physical_params[PHYSICAL_PARAM_WHEELBASE];
    const double max_steering = physical_params[PHYSICAL_PARAM_MAX_STEERING_RAD];
    const int width = params[FRAME_PARAM_WIDTH];
    const int height = params[FRAME_PARAM_HEIGHT];

    const float heading = -goal_heading;
    const float steering = positive_steer ? max_steering : -max_steering;
    const float tan_steering = tanf(steering);
    const float beta = atanf(0.5 * tan_steering);
    const float heading_increment_factor = (velocity_px_s * cosf(beta) * tan_steering) / (2 * lr);

    float xp = TO_FLOAT(goal.x), zp = TO_FLOAT(goal.y);
}