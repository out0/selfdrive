#include "../include/cuda_basic.h"
#include "../include/class_def.h"

__global__ static void __CUDA_KERNEL_FrameColor(float3 *frame, uchar3 *output, int width, int height, uchar3 *classColors)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int y = pos / width;
    int x = pos - y * width;

    if (y >= height)
        return;
    if (x >= width)
        return;

    int segClass = frame[pos].x;

    output[pos].x = classColors[segClass].x;
    output[pos].y = classColors[segClass].y;
    output[pos].z = classColors[segClass].z;
}


uchar3 *CUDA_convertFrameColors(float3 *frame, int width, int height)
{
    const int numClasses = 29;

    uchar3 *resultImgPtr = nullptr;
    if (!cudaAllocMapped(&resultImgPtr, sizeof(uchar3) * (width * height)))
        return nullptr;

    uchar3 *classColors;
    if (!cudaAllocMapped(&classColors, sizeof(uchar3) * numClasses))
    {
        cudaFreeHost(resultImgPtr);
        return nullptr;
    }

    for (int i = 0; i < numClasses; i++)
    {
        classColors[i].x = segmentationClassColors[i][0];
        classColors[i].y = segmentationClassColors[i][1];
        classColors[i].z = segmentationClassColors[i][2];
    }

    int size = width * height;
    int numBlocks = floor(size / 256) + 1;

    __CUDA_KERNEL_FrameColor<<<numBlocks, 256>>>(frame, resultImgPtr, width, height, classColors);

    CUDA(cudaDeviceSynchronize());
    cudaFreeHost(classColors);
    return resultImgPtr;
}