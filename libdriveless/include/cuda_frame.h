#pragma once

#ifndef __CUDA_FRAME_DRIVELESS_H
#define __CUDA_FRAME_DRIVELESS_H

#include "cuda_basic.h"
#include "cuda_ptr.h"
// CODE:BEGIN

#include <stdexcept>

#if defined(CUDA_VERSION_MAJOR) && CUDA_VERSION_MAJOR >= 13
#define double4 double4_16a
#endif

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
class CudaFrame
{
private:
    cptr<T> frame;
    const int _width;
    const int _height;

    static void copyData(float *ptr, T *dest, long pos);
    //static void copyBackData(T *orig, float *dest, long pos);

protected:
    T &at(std::pair<size_t, size_t> indices)
    {
        if (indices.first >= _width || indices.second >= _height)
        {
            throw std::out_of_range("Index out of bounds!");
        }
        long pos = indices.second * _width + indices.first;
        return frame->get()[pos];
    }

public:
    CudaFrame(int width, int height);

    virtual void copyFrom(float *ptr);
    virtual void clear();
    inline T *getCudaPtr() { return frame->get(); }

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
CudaFrame<T>::CudaFrame(int width, int height) : _width(width), _height(height)
{
    size_t size = _width * _height;
    this->frame = std::make_unique<CudaPtr<T>>(size);
}

template <typename T>
void CudaFrame<T>::copyFrom(float *ptr)
{
    for (int i = 0; i < _height; i++)
        for (int j = 0; j < _width; j++)
        {
            long pos = (_width * i + j);
            CudaFrame<T>::copyData(ptr, frame->get(), pos);
        }
}

template <typename T>
void CudaFrame<T>::clear()
{
    CUDA_clear(frame->get(), _width, _height);
}

template <typename T>
void CudaFrame<T>::copyData(float *ptr, T *dest, long pos)
{
    dest[pos] = static_cast<T>(ptr[pos]);
}

template <>
inline void CudaFrame<float4>::copyData(float *ptr, float4 *dest, long pos)
{
    long posPtr = 4 * pos;
    dest[pos].x = static_cast<float>(ptr[posPtr]);
    dest[pos].y = static_cast<float>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<float>(ptr[posPtr + 2]);
    dest[pos].w = static_cast<float>(ptr[posPtr + 3]);
}
template <>
inline void CudaFrame<double4>::copyData(float *ptr, double4 *dest, long pos)
{
    long posPtr = 4 * pos;
    dest[pos].x = static_cast<double>(ptr[posPtr]);
    dest[pos].y = static_cast<double>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<double>(ptr[posPtr + 2]);
    dest[pos].w = static_cast<double>(ptr[posPtr + 3]);
}

template <>
inline void CudaFrame<int4>::copyData(float *ptr, int4 *dest, long pos)
{
    long posPtr = 4 * pos;
    dest[pos].x = static_cast<int>(ptr[posPtr]);
    dest[pos].y = static_cast<int>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<int>(ptr[posPtr + 2]);
    dest[pos].w = static_cast<int>(ptr[posPtr + 3]);
}
template <>
inline void CudaFrame<float3>::copyData(float *ptr, float3 *dest, long pos)
{
    long posPtr = 3 * pos;
    dest[pos].x = static_cast<float>(ptr[posPtr]);
    dest[pos].y = static_cast<float>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<float>(ptr[posPtr + 2]);
}
template <>
inline void CudaFrame<double3>::copyData(float *ptr, double3 *dest, long pos)
{
    long posPtr = 3 * pos;
    dest[pos].x = static_cast<double>(ptr[posPtr]);
    dest[pos].y = static_cast<double>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<double>(ptr[posPtr + 2]);
}
template <>
inline void CudaFrame<int3>::copyData(float *ptr, int3 *dest, long pos)
{
    long posPtr = 3 * pos;
    dest[pos].x = static_cast<int>(ptr[posPtr]);
    dest[pos].y = static_cast<int>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<int>(ptr[posPtr + 2]);
}
template <>
inline void CudaFrame<float2>::copyData(float *ptr, float2 *dest, long pos)
{
    long posPtr = 2 * pos;
    dest[pos].x = static_cast<float>(ptr[posPtr]);
    dest[pos].y = static_cast<float>(ptr[posPtr + 1]);
}
template <>
inline void CudaFrame<double2>::copyData(float *ptr, double2 *dest, long pos)
{
    long posPtr = 2 * pos;
    dest[pos].x = ptr[posPtr];
    dest[pos].y = ptr[posPtr + 1];
}
template <>
inline void CudaFrame<int2>::copyData(float *ptr, int2 *dest, long pos)
{
    long posPtr = 2 * pos;
    dest[pos].x = static_cast<int>(ptr[posPtr]);
    dest[pos].y = static_cast<int>(ptr[posPtr + 1]);
}


#endif
