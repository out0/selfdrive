#include "../../include/cuda_params.h"
#include "../../include/math_utils.h"

__device__ __host__ double getFrameCostCuda(float3 *frame, float *classCost, long pos) {
    int segmentation_class = TO_INT(frame[pos].x);
    return classCost[segmentation_class];
}

#include <stdio.h>

__device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int x, int z, float angle_radians)
{

    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int min_dist_x = params[FRAME_PARAM_MIN_DIST_X];
    int min_dist_z = params[FRAME_PARAM_MIN_DIST_Z];
    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    float c = cosf(angle_radians);
    float s = sinf(angle_radians);

    for (int i = -min_dist_z; i <= min_dist_z; i++)
        for (int j = -min_dist_x; j <= min_dist_x; j++)
        {
            int xl = TO_INT(j * c - i * s + x);
            int zl = TO_INT(j * s + i * c + z);

            if (xl < 0 || xl >= width)
                continue;

            if (zl < 0 || zl >= height)
                continue;

            if (xl >= lower_bound_ego_x && xl <= upper_bound_ego_x && zl >= upper_bound_ego_z && zl <= lower_bound_ego_z)
                continue;

            double cost = getFrameCostCuda(frame, classCost, zl * width + xl);

            if (cost < 0)
            {
                // if (x == 128 && z == 127)
                // {
                //     printf("(%d, %d) not feasible on angle %f because of position: (%d, %d)\n", x, z, angle_radians * 180 / PI, xl, zl);
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