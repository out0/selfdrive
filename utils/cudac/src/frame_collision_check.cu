#include "../include/cuda_basic.h"
#include "../include/class_def.h"
#include <math_constants.h>

__device__ bool __CUDA_KERNEL_ComputeFeasibleForAngle(
    float3 *frame,
    int *classCost,
    int x,
    int z,
    float angle_radians,
    int width,
    int height,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_ego_x,
    int lower_bound_ego_z,
    int upper_bound_ego_x,
    int upper_bound_ego_z)
{
    float c = cosf(angle_radians);
    float s = sinf(angle_radians);

    for (int j = -min_dist_z; j <= min_dist_z; j++)
        for (int i = -min_dist_x; i <= min_dist_x; i++)
        {
            int xl = __float2int_rn(j * c - i * s + x);
            int zl = __float2int_rn(j * s + i * c + z);

            if (xl < 0 || xl >= width)
                continue;

            if (zl < 0 || zl >= height)
                continue;

            if (xl >= lower_bound_ego_x && xl <= upper_bound_ego_x && zl >= upper_bound_ego_z && zl <= lower_bound_ego_z)
                continue;

            int segmentation_class = __float2int_rn(frame[zl * width + xl].x);

            if (classCost[segmentation_class] < 0)
            {
                // if (x == 115 && z == 16)
                // {
                //     printf("(%d, %d) not feasible on angle %f because of position: (%d, %d)\n", x, z, angle_radians * 180 / CUDART_PI_F, xl, zl);
                //     printf("(%d, %d) min distances: W: %d  H: %d\n",  x, z, min_dist_x, min_dist_z);
                // }
                return false;
            }
        }

    // if (x == 115 && z == 16)
    // {
    //     printf("(%d, %d) feasible on angle %f\n", x, z, angle_radians * 180 / CUDART_PI_F);
    // }
    return true;
}
