#include "../include/search_frame.h"



__device__ __host__ void count_obstacle_in_search_zones(float3 *frame, float *classCosts, int *search_params, int2 search_zone_dim, uint2 *search_zone_info, int pos)
{
    const int width = search_params[FRAME_PARAM_WIDTH];

    int z = pos / width;
    int x = pos - z * width;

    const int WG = search_zone_dim.x;
    const int HG = search_zone_dim.y;

    int xg = x / WG;
    int zg = z / HG;

    int segmentation_class = TO_INT(frame[pos].x);

    if (classCosts[segmentation_class] < 0)
    {
        int posg = zg * search_zone_dim.x + xg;
        const int A = xg * WG;
        const int B = zg * HG;

#if defined(DRIVELESS_CUDA_ENABLED) && defined(__CUDA_ARCH__)
        atomicInc(&search_zone_info[posg].x, 99999999);
        // computes an obstacle on the borders
        if ((x >= A && x <= (A + WG - 1)) && (z == B || z == (B + HG - 1)))
            atomicInc(&search_zone_info[posg].y, 99999999);

        else if ((z >= B && z <= (B + HG - 1)) && (x == A || x == (A + WG - 1)))
            atomicInc(&search_zone_info[posg].y, 99999999);
#else
        __atomic_fetch_add(&search_zone_info[posg].x, 1, __ATOMIC_SEQ_CST);
        if ((x >= A && x <= (A + WG - 1)) && (z == B || z == (B + HG - 1)))
            __atomic_fetch_add(&search_zone_info[posg].y, 1, __ATOMIC_SEQ_CST);
        else if ((z >= B && z <= (B + HG - 1)) && (x == A || x == (A + WG - 1)))
            __atomic_fetch_add(&search_zone_info[posg].y, 1, __ATOMIC_SEQ_CST);
#endif
    }
}