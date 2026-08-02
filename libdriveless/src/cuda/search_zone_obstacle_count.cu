#include "../../include/search_frame.h"

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

#define SEARCH_ZONE_ID(grid_width, xg, zg) (zg * grid_width + xg)

int2 SearchFrame::getSearchZoneLocation(int x, int z)
{
    return zone_location({_searchZoneDim.first, _searchZoneDim.second},
                         {_search_zone_info->width(), _search_zone_info->height()}, x, z);
}

int SearchFrame::getSearchZoneId(int x, int z)
{
    int2 addr = getSearchZoneLocation(x, z);
    return SEARCH_ZONE_ID(_search_zone_info->width(), addr.x, addr.y);
}

__device__ __host__ bool is_zone_border(int x, int z, int xg, int zg, int search_zone_dim_w, int search_zone_dim_h) {
    const int A = xg * search_zone_dim_w;
    const int B = zg * search_zone_dim_h;

    return (x >= A && x <= (A + search_zone_dim_w - 1)) && (z == B || z == (B + search_zone_dim_h - 1)) || 
        (z >= B && z <= (B + search_zone_dim_h - 1)) && (x == A || x == (A + search_zone_dim_w - 1));
}

__device__ __host__ void count_obstacle_in_search_zones(float3 *frame, float *classCosts, int *search_params, uint2 *search_zone_info, int pos)
{
    const int width = search_params[FRAME_PARAM_WIDTH];

    const int search_zone_dim_w = search_params[FRAME_SEARCH_ZONE_DIM_WIDTH];
    const int search_zone_dim_h = search_params[FRAME_SEARCH_ZONE_DIM_HEIGHT];
    const int search_zone_grid_w = search_params[FRAME_SEARCH_ZONE_GRID_WIDTH];
    const int search_zone_grid_h = search_params[FRAME_SEARCH_ZONE_GRID_HEIGHT];

    int z = pos / width;
    int x = pos - z * width;

    int2 location = zone_location({search_zone_dim_w, search_zone_dim_h}, {search_zone_grid_w, search_zone_grid_h}, x, z);

    int segmentation_class = TO_INT(frame[pos].x);

    if (classCosts[segmentation_class] < 0)
    {
        int posg = SEARCH_ZONE_ID(search_zone_grid_w, location.x, location.y);

#if defined(DRIVELESS_CUDA_ENABLED) && defined(__CUDA_ARCH__)
        atomicInc(&search_zone_info[posg].x, 99999999);
        // computes an obstacle on the borders
        if (is_zone_border(x, z, location.x, location.y, search_zone_dim_w, search_zone_dim_h))
            atomicInc(&search_zone_info[posg].y, 99999999);
#else
        __atomic_fetch_add(&search_zone_info[posg].x, 1, __ATOMIC_SEQ_CST);
        if (is_zone_border(x, z, location.x, location.y, search_zone_dim_w, search_zone_dim_h))
            __atomic_fetch_add(&search_zone_info[posg].y, 1, __ATOMIC_SEQ_CST);
#endif
    }
}

