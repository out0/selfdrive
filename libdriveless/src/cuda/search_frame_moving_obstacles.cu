#include "../../include/search_frame.h"
#include "../../include/cuda_basic.h"
#include "../../include/moving_obstacle.h"
#include <stdexcept>

extern __global__ void __CUDA_propagate_moving_obstacles(float3 *frame, int *params, float *obstacles, int count)
{
    int width = params[FRAME_PARAM_WIDTH];
    int height = params[FRAME_PARAM_HEIGHT];

    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos > width * height)
        return;

    // int z = pos / width;
    // int x = pos - z * width;

    // int i = 0;
    // while (i < count) {

    // }
}

void SearchFrame::setMovingObstacles(std::vector<MovingObstacle> obstacles)
{
    auto p = MovingObstacle::listToCuda(obstacles);

    int size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    int mx = _params->get()[FRAME_PARAM_MIN_DIST_X];
    int mz = _params->get()[FRAME_PARAM_MIN_DIST_Z];

    int half_minDist_px = TO_INT(0.5 * sqrtf(mx * mx + mz * mz));

    __CUDA_propagate_moving_obstacles<<<numBlocks, THREADS_IN_BLOCK>>>(getCudaPtr(), _params->get(), p->get(), obstacles.size() );
    CUDA(cudaDeviceSynchronize());
}