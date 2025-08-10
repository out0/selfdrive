#include "../../include/search_frame.h"
#include "../../include/cuda_basic.h"
#include <stdexcept>
#include <tuple>

__global__ static void __CUDA_KERNEL_FrameColor(float3 *frame, uchar3 *output, int width, int height, uchar3 *classColors, int classCount);

void SearchFrame::setClassColors(std::vector<std::tuple<int, int, int>> colors)
{   
    if (colors.size() == 0)
        return;

    if (_classCount > 0 && colors.size() != _classCount) {
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

bool SearchFrame::exportToColorFrame(uchar *dest)
{
    if (_classColors == nullptr)
        return false;

    uchar3 *resultImgPtr = nullptr;
    if (!cudaAllocMapped(&resultImgPtr, sizeof(uchar3) * (width() * height())))
        return false;

    int size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_FrameColor<<<numBlocks, THREADS_IN_BLOCK>>>(getCudaPtr(), resultImgPtr, width(), height(), _classColors->get(), _classCount);

    CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < size; i++) {
        long pos = 3 * i;
        dest[pos] = resultImgPtr[i].x;
        dest[pos+1] = resultImgPtr[i].y;
        dest[pos+2] = resultImgPtr[i].z;
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
    if (segClass < 0 || segClass >= classCount) return;

    output[pos].x = classColors[segClass].x;
    output[pos].y = classColors[segClass].y;
    output[pos].z = classColors[segClass].z;
}