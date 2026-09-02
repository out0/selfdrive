#include <driveless/math_utils.h>
#include "wpmp_data.h"

__device__ __host__ bool is_feasible_for_all_angles(float3 *frame, long pos)
{
    return TO_INT(frame[pos].z) & 256 > 0;
}

__device__ __host__ float traversability_cost(float3 *frame, int *params, float *classCost, int2 min_distance, int x, int z, float angle_radians)
{
    const int width = params[FRAME_PARAM_WIDTH];
    const int height = params[FRAME_PARAM_HEIGHT];
    const long pos = COMPUTE_POS(width, x, z);
    const bool pre_process_collision_vector = params[FRAME_PREPROCESS_COLLISION_TYPE] == PREPROCESS_COLLISION_VECTOR;
    const bool pre_process_collision_dist = pre_process_collision_vector || params[FRAME_PREPROCESS_COLLISION_TYPE] == PREPROCESS_COLLISION_DIST;

    if (pre_process_collision_dist)
    {
        if (is_feasible_for_all_angles(frame, pos))
        {
            int segmentation_class = TO_INT(frame[COMPUTE_POS(width, x, z)].x);
            return classCost[segmentation_class];
        }
    } else {
         printf("pre_process_collision_dist is false\n");
    }

    const int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    const int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    const int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    const int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    float c = cosf(angle_radians);
    float s = sinf(angle_radians);

    for (int i = -min_distance.y; i <= min_distance.y; i++)
        for (int j = -min_distance.x; j <= min_distance.x; j++)
        {
            int xl = TO_INT(j * c - i * s + x);
            int zl = TO_INT(j * s + i * c + z);

            if (xl < 0 || xl >= width)
                continue;

            if (zl < 0 || zl >= height)
                continue;

            if (xl >= lower_bound_ego_x && xl <= upper_bound_ego_x && zl >= upper_bound_ego_z && zl <= lower_bound_ego_z)
                continue;

            int segmentation_class = TO_INT(frame[COMPUTE_POS(width, xl, zl)].x);
            float cost = classCost[segmentation_class];
            if (cost < 0)
                return -1;
        }

    int segmentation_class = TO_INT(frame[COMPUTE_POS(width, x, z)].x);
    return classCost[segmentation_class];
}
