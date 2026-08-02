#include "../../include/search_frame.h"
#include "../../include/cuda_basic.h"
#include <stdexcept>
#include <tuple>

__global__ static void __CUDA_KERNEL_FrameColor(float3 *frame, uchar3 *output, int width, int height, uchar3 *classColors, int classCount);
extern __device__ __host__ bool is_zone_border(int x, int z, int xg, int zg, int search_zone_dim_w, int search_zone_dim_h);
extern __device__ __host__ int2 zone_location(int2 zone_dim_size, int2 zone_grid_size, int x, int z);

__global__ static void __CUDA_KERNEL_ShowZoneMarks(float3 *frame, uchar3 *output, int width, int height, int *search_params)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int z = pos / width;
    int x = pos - z * width;

    if (z >= height)
        return;
    if (x >= width)
        return;

    const int search_zone_dim_w = search_params[FRAME_SEARCH_ZONE_DIM_WIDTH];
    const int search_zone_dim_h = search_params[FRAME_SEARCH_ZONE_DIM_HEIGHT];
    const int search_zone_grid_w = search_params[FRAME_SEARCH_ZONE_GRID_WIDTH];
    const int search_zone_grid_h = search_params[FRAME_SEARCH_ZONE_GRID_HEIGHT];

    int2 location = zone_location({search_zone_dim_w, search_zone_dim_h}, {search_zone_grid_w, search_zone_grid_h}, x, z);

    if (is_zone_border(x, z, location.x, location.y, search_zone_dim_w, search_zone_dim_h))
    {
        output[pos].x = 128;
        output[pos].y = 128;
        output[pos].z = 128;
    }
}



void SearchFrame::setClassColors(std::vector<std::tuple<int, int, int>> colors)
{
    if (_classCount > 0 && colors.size() != _classCount)
    {
        throw std::invalid_argument("invalid number of classed on setClassColors(). Expected: " + std::to_string(_classCount) + " obtained: " + std::to_string(colors.size()));
    }

    if (colors.size() == 0)
        return;

    if (_classCount > 0 && colors.size() != _classCount)
    {
        throw std::invalid_argument("invalid number of classed on setClassColors(). Expected: " + std::to_string(_classCount) + " obtained: " + std::to_string(colors.size()));
    }

    _classCount = colors.size();
    _classColors = std::make_unique<CudaPtr<uchar3>>(colors.size());

    int i = 0;
    for (auto const &c : colors)
    {
        std::tie(_classColors->get()[i].x, _classColors->get()[i].y, _classColors->get()[i].z) = c;
        i++;
    }
}

bool SearchFrame::exportToColorFrame(uchar *dest, bool show_search_zone_marks)
{
    if (_classColors == nullptr)
        return false;

    uchar3 *resultImgPtr = nullptr;
    if (!cudaAllocMapped(&resultImgPtr, sizeof(uchar3) * (width() * height())))
        return false;

    int size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_FrameColor<<<numBlocks, THREADS_IN_BLOCK>>>(getPtr(), resultImgPtr, width(), height(), _classColors->get(), _classCount);
    CUDA(cudaDeviceSynchronize());

    if (show_search_zone_marks)
    {
        __CUDA_KERNEL_ShowZoneMarks<<<numBlocks, THREADS_IN_BLOCK>>>(getPtr(), resultImgPtr, width(), height(), _params->get());
        CUDA(cudaDeviceSynchronize());
    }

    for (int i = 0; i < size; i++)
    {
        long pos = 3 * i;
        dest[pos] = resultImgPtr[i].x;
        dest[pos + 1] = resultImgPtr[i].y;
        dest[pos + 2] = resultImgPtr[i].z;
    }

    cudaFreeHost(resultImgPtr);
    return true;
}

__global__ static void __CUDA_KERNEL_FrameColor(float3 *frame, uchar3 *output, int width, int height, uchar3 *classColors, int classCount)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int y = pos / width;
    int x = pos - y * width;

    if (y >= height)
        return;
    if (x >= width)
        return;

    int segClass = frame[pos].x;
    if (segClass < 0 || segClass >= classCount)
        return;

    output[pos].x = classColors[segClass].x;
    output[pos].y = classColors[segClass].y;
    output[pos].z = classColors[segClass].z;
}
