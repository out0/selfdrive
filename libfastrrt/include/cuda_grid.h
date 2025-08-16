#pragma once

#ifndef __CUDA_GRID_DRIVELESS_H
#define __CUDA_GRID_DRIVELESS_H


#include <driveless/cuda_basic.h>

// CODE:BEGIN

#if defined(CUDA_VERSION_MAJOR) && CUDA_VERSION_MAJOR >= 13
#define double4 double4_16a
#endif

#include <stdexcept>
extern void CUDA_clear(double4 *frame, int width, int height);
extern void CUDA_clear(float4 *frame, int width, int height);
extern void CUDA_clear(int4 *frame, int width, int height);
extern void CUDA_clear(double3 *frame, int width, int height);
extern void CUDA_clear(float3 *frame, int width, int height);
extern void CUDA_clear(int3 *frame, int width, int height);
extern void CUDA_clear(double2 *frame, int width, int height);
extern void CUDA_clear(float2 *frame, int width, int height);
extern void CUDA_clear(int2 *frame, int width, int height);
extern void CUDA_clear(double *frame, int width, int height);
extern void CUDA_clear(float *frame, int width, int height);
extern void CUDA_clear(int *frame, int width, int height);

template <typename T>
class CudaGrid
{
private:
    T *frame;
    const int _width;
    const int _height;

    static void copyData(float *ptr, T *dest, long pos);

protected:
    T &at(std::pair<size_t, size_t> indices)
    {
        if (indices.first >= _width || indices.second >= _height)
        {
            throw std::out_of_range("Index out of bounds!");
        }
        long pos = indices.second * _width + indices.first;
        return frame[pos];
    }

public:
    CudaGrid(int width, int height);
    ~CudaGrid();

    virtual void copyFrom(float *ptr);
    virtual void clear();
    inline T *getCudaPtr() { return frame; }

    constexpr int width()
    {
        return _width;
    }
    constexpr int height()
    {
        return _height;
    }

    T &operator[](std::pair<size_t, size_t> indices)
    {
        return at(indices);
    }

    T &operator[](long pos)
    {
        long p = _width * _height;
        if (pos > p)
        {
            throw std::out_of_range("Index out of bounds!");
        }
        return frame[pos];
    }
};

template <typename T>
CudaGrid<T>::CudaGrid(int width, int height) : _width(width), _height(height)
{
    size_t size = sizeof(T) * (_width * _height);

    if (!cudaAllocMapped(&this->frame, size))
    {
        fprintf(stderr, "[CUDA FRAME] unable to allocate frame memory with %ld bytes\n", size);
        this->frame = nullptr;
        return;
    }
}

template <typename T>
CudaGrid<T>::~CudaGrid()
{
    cudaFreeHost(frame);
}

template <typename T>
void CudaGrid<T>::copyFrom(float *ptr)
{
    for (int i = 0; i < _height; i++)
        for (int j = 0; j < _width; j++)
        {
            long pos = (_width * i + j);
            CudaGrid<T>::copyData(ptr, frame, pos);
        }
}

template <typename T>
void CudaGrid<T>::clear()
{
    CUDA_clear(frame, _width, _height);
}

template <typename T>
void CudaGrid<T>::copyData(float *ptr, T *dest, long pos)
{
    dest[pos] = static_cast<T>(ptr[pos]);
}

template <>
inline void CudaGrid<float4>::copyData(float *ptr, float4 *dest, long pos)
{
    long posPtr = 4 * pos;
    dest[pos].x = static_cast<float>(ptr[posPtr]);
    dest[pos].y = static_cast<float>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<float>(ptr[posPtr + 2]);
    dest[pos].w = static_cast<float>(ptr[posPtr + 3]);
}
template <>
inline void CudaGrid<double4>::copyData(float *ptr, double4 *dest, long pos)
{
    long posPtr = 4 * pos;
    dest[pos].x = static_cast<double>(ptr[posPtr]);
    dest[pos].y = static_cast<double>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<double>(ptr[posPtr + 2]);
    dest[pos].w = static_cast<double>(ptr[posPtr + 3]);
}
template <>
inline void CudaGrid<int4>::copyData(float *ptr, int4 *dest, long pos)
{
    long posPtr = 4 * pos;
    dest[pos].x = static_cast<int>(ptr[posPtr]);
    dest[pos].y = static_cast<int>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<int>(ptr[posPtr + 2]);
    dest[pos].w = static_cast<int>(ptr[posPtr + 3]);
}
template <>
inline void CudaGrid<float3>::copyData(float *ptr, float3 *dest, long pos)
{
    long posPtr = 3 * pos;
    dest[pos].x = static_cast<float>(ptr[posPtr]);
    dest[pos].y = static_cast<float>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<float>(ptr[posPtr + 2]);
}
template <>
inline void CudaGrid<double3>::copyData(float *ptr, double3 *dest, long pos)
{
    long posPtr = 3 * pos;
    dest[pos].x = static_cast<double>(ptr[posPtr]);
    dest[pos].y = static_cast<double>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<double>(ptr[posPtr + 2]);
}
template <>
inline void CudaGrid<int3>::copyData(float *ptr, int3 *dest, long pos)
{
    long posPtr = 3 * pos;
    dest[pos].x = static_cast<int>(ptr[posPtr]);
    dest[pos].y = static_cast<int>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<int>(ptr[posPtr + 2]);
}
template <>
inline void CudaGrid<float2>::copyData(float *ptr, float2 *dest, long pos)
{
    long posPtr = 2 * pos;
    dest[pos].x = static_cast<float>(ptr[posPtr]);
    dest[pos].y = static_cast<float>(ptr[posPtr + 1]);
}
template <>
inline void CudaGrid<double2>::copyData(float *ptr, double2 *dest, long pos)
{
    long posPtr = 2 * pos;
    dest[pos].x = ptr[posPtr];
    dest[pos].y = ptr[posPtr + 1];
}
template <>
inline void CudaGrid<int2>::copyData(float *ptr, int2 *dest, long pos)
{
    long posPtr = 2 * pos;
    dest[pos].x = static_cast<int>(ptr[posPtr]);
    dest[pos].y = static_cast<int>(ptr[posPtr + 1]);
}

// CODE:END

#endif
