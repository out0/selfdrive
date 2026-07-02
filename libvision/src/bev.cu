#include "../include/bev.h"
#include <driveless/math_utils.h>

__global__ void __CUDA_compute_birds_view(float3 *restPtr,
                                          float3 *frontPtr,
                                          float3 *backPtr,
                                          float3 *leftPtr,
                                          float3 *rightPtr,
                                          int width,
                                          int height,
                                          int carSizeW,
                                          int carSizeH,
                                          int selfClassCode);

BEV::BEV(int width, int height, std::pair<int, int> carSizePx, int carClassCode)
{
    _width = width;
    _height = height;
    _data = new SearchFrame(width, height, {-1, -1}, {-1, -1});
    _carSizePx = carSizePx;
    _carClassCode = carClassCode;
}

void BEV::compute(
    SearchFrame *front,
    SearchFrame *back,
    SearchFrame *left,
    SearchFrame *right)
{
    float3 *frontPtr = front->getPtr();
    float3 *backPtr = back->getPtr();
    float3 *leftPtr = left->getPtr();
    float3 *rightPtr = right->getPtr();
    float3 *resPtr = _data->getPtr();

    int carSizeW = _carSizePx.first,
        carSizeH = _carSizePx.second;

    int size = _width * _height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;
    __CUDA_compute_birds_view<<<numBlocks, THREADS_IN_BLOCK>>>(resPtr, frontPtr, backPtr, leftPtr, rightPtr, _width, _height, carSizeW, carSizeH, _carClassCode);
    CUDA(cudaDeviceSynchronize());
}

__global__ void __CUDA_compute_birds_view(float3 *restPtr,
                                          float3 *frontPtr,
                                          float3 *backPtr,
                                          float3 *leftPtr,
                                          float3 *rightPtr,
                                          int width,
                                          int height,
                                          int carSizeW,
                                          int carSizeH,
                                          int selfClassCode)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int z = pos / width;
    int x = pos - z * width;

    if (z >= height)
        return;
    if (x >= width)
        return;

    int x_center = TO_INT(width / 2);
    int z_center = TO_INT(height / 2);
    int half_car_w = TO_INT(carSizeW / 2);
    int half_car_h = TO_INT(carSizeH / 2);

    if ((abs(x - x_center) <= half_car_w) &&
        (abs(z - z_center) <= half_car_h))
        restPtr[pos].x = selfClassCode;
}