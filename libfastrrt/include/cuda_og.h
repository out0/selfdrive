#pragma once

#ifndef __CUDA_OG_DRIVELESS_H
#define __CUDA_OG_DRIVELESS_H

#include "../../cudac/include/cuda_basic.h"
#include "angle.h"
//#include <stdexcept>

class CudaOccupancyGrid
{
private:
    float3 *frame;
    float *_classCosts;
    const int _width;
    const int _height;
    const int _minDistanceX_px;
    const int _minDistanceZ_px;
    const int2 _lowerBound;
    const int2 _upperBound;

public:
    CudaOccupancyGrid(
        float3 *ptr, 
        float *classCosts,
        int width, 
        int height,
        int minDistanceX_px,
        int minDistanceZ_px,
        int2 lowerBound,
        int2 upperBound);
    ~CudaOccupancyGrid();

    inline float3 *getCudaPtr() { return frame; }

    constexpr int width()
    {
        return _width;
    }
    constexpr int height()
    {
        return _height;
    }

    void computeBoundaries();
};


#endif // __CUDA_OG_DRIVELESS_H
