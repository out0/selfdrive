#include "../include/search_frame.h"

extern __device__ __host__ void count_obstacle_in_search_zones(float3 *frame, float *classCosts, int *search_params, int2 search_zone_dim, uint2 *search_zone_info, int pos);

void SearchFrame::initialize_search_zones()
{
    int W = TO_INT(round(static_cast<double>(width()) / static_cast<double>(_searchZoneDim.first))) + 1;
    int H = TO_INT(round(static_cast<double>(height()) / static_cast<double>(_searchZoneDim.second))) + 1;
    _search_zone_info = std::make_unique<Frame<uint2>>(W, H, _numCPUThreadHandlers);
}

__global__ void pre_compute_search_zones_cuda(float3 *frame, float *classCosts, int *search_params, int2 search_zone_dim, uint2 *search_zone_info)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int width = search_params[FRAME_PARAM_WIDTH];
    int height = search_params[FRAME_PARAM_HEIGHT];

    if (pos >= width * height)
        return;

    count_obstacle_in_search_zones(frame, classCosts, search_params, search_zone_dim, search_zone_info, pos);
}

void SearchFrame::pre_compute_search_zones()
{
    int size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    int2 search_zone_dim = {_searchZoneDim.first, _searchZoneDim.second};

    pre_compute_search_zones_cuda<<<numBlocks, THREADS_IN_BLOCK>>>(getPtr(), _classCosts->get(), _params->get(), search_zone_dim, _search_zone_info->getPtr());

    CUDA(cudaDeviceSynchronize());
}