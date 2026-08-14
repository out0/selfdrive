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

__device__ __host__ int4 sz_egdes_frame_pos(int2 sz_location, int2 search_zone_dim, int frame_width) {
    const int A = sz_location.x * search_zone_dim.x;
    const int B = sz_location.y * search_zone_dim.y;
    const int tl = COMPUTE_POS(frame_width, A, B);
    const int tr = COMPUTE_POS(frame_width, (A + search_zone_dim.x - 1), B);
    const int bl = COMPUTE_POS(frame_width, A, (B + search_zone_dim.y - 1));
    const int br = COMPUTE_POS(frame_width, (A + search_zone_dim.x - 1), (B + search_zone_dim.y - 1));
    return {tl, tr, bl, br};
}

