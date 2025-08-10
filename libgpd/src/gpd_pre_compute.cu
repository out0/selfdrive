#include <stdexcept>
#include "../include/gpd.h"

#define MAX_COST 9999999

inline __device__ int compute_pos(int width, int height, int x, int z)
{
    if (x < 0 || z < 0 || x >= width || z >= height)  return -1;
    return width * z + x;
}

__global__ void __CUDA_precompute_exclusion_zone(float3 *frame, int *params, float *classCost, float heading)
{
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int max = width * height;
    if (pos > max)
        return;

    int z = pos / width;
    int x = pos - z * width;

    int lower_bound_ego_x = params[FRAME_PARAM_LOWER_BOUND_X];
    int lower_bound_ego_z = params[FRAME_PARAM_LOWER_BOUND_Z];
    int upper_bound_ego_x = params[FRAME_PARAM_UPPER_BOUND_X];
    int upper_bound_ego_z = params[FRAME_PARAM_UPPER_BOUND_Z];

    if (x >= lower_bound_ego_x && x <= upper_bound_ego_x && z >= upper_bound_ego_z && z <= lower_bound_ego_z)
    {
        frame[pos].y = MAX_COST;
        return;
    }

    if (classCost[(int)frame[pos].x] < 0 || ((int)frame[pos].z & 0x100) <= 0)
    {
        float c = cosf(heading);
        float s = sinf(heading);

        float xl = x;
        float zl = z;

        while (xl < width && zl < height) {
            int pos_propagate = compute_pos(width, height, TO_INT(xl), TO_INT(zl));
            if (pos_propagate < 0 || pos_propagate >= max) break;
            if (classCost[(int)frame[pos_propagate].x] < 0 || (int)frame[pos_propagate].z & 0x100 <= 0)
                break;
            frame[pos_propagate].z = (int)frame[pos_propagate].z | 0x200;
            xl = xl - s;
            zl = zl + c;
        }
        
        xl = x;
        zl = z;
        while (xl >= 0 && zl >= 0) {
            int pos_propagate = compute_pos(width, height, TO_INT(xl), TO_INT(zl));
            if (pos_propagate < 0 || pos_propagate >= max) break;
            if (classCost[(int)frame[pos_propagate].x] < 0 || (int)frame[pos_propagate].z & 0x100 <= 0)
                return;
            frame[pos_propagate].z = (int)frame[pos_propagate].z | 0x200;
            xl = xl + s;
            zl = zl - c;
        }

        // for (int zl = (z + 1); zl < height; zl++)
        // {
        //     int pos_propagate = compute_pos(width, x, zl);

        //     if (classCost[(int)frame[pos_propagate].x] < 0 || (int)frame[pos_propagate].z & 0x100 <= 0)
        //         return;

        //     frame[pos_propagate].z = (int)frame[pos_propagate].z | 0x200;
        // }
        
    }
}

void GoalPointDiscover::computeExclusionZone(SearchFrame &frame, angle heading)
{
    size_t size = frame.width() * frame.height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    if (!frame.isSafeZoneChecked()) {
        throw std::runtime_error("Safe zone check must be performed before computing exclusion zone.");
    }
    __CUDA_precompute_exclusion_zone<<<numBlocks, THREADS_IN_BLOCK>>>(frame.getCudaPtr(), frame.getCudaFrameParamsPtr(), frame.getCudaClassCostsPtr(), heading.rad());
    CUDA(cudaDeviceSynchronize());
}