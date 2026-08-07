#pragma once

#ifndef __CUDA_FRAME_DRIVELESS_H
#define __CUDA_FRAME_DRIVELESS_H

#include "cuda_basic.h"
#include "cpu_parallel_copy.h"
#include <stdexcept>

#include "driveless_config.h"

#ifdef DRIVELESS_CUDA_ENABLED


extern void CUDA_clear(DOUBLE4 *frame, int width, int height);
extern void CUDA_clear(float4 *frame, int width, int height);
extern void CUDA_clear(int4 *frame, int width, int height);
extern void CUDA_clear(double3 *frame, int width, int height);
extern void CUDA_clear(float3 *frame, int width, int height);
extern void CUDA_clear(int3 *frame, int width, int height);
extern void CUDA_clear(double2 *frame, int width, int height);
extern void CUDA_clear(float2 *frame, int width, int height);
extern void CUDA_clear(int2 *frame, int width, int height);
extern void CUDA_clear(uint4 *frame, int width, int height);
extern void CUDA_clear(double *frame, int width, int height);
extern void CUDA_clear(float *frame, int width, int height);
extern void CUDA_clear(int *frame, int width, int height);
#else
#include <memory>
#include "cpu_parallel_clear.h"
#endif

template <typename T>
class Frame
{
private:
#ifdef DRIVELESS_CUDA_ENABLED
    cptr<T> _frame;
#else
    std::unique_ptr<T[]> _frame;
#endif
    const int _width;
    const int _height;
    const int _numCPUThreadHandlers;

protected:
    T &at(std::pair<size_t, size_t> indices)
    {
        if (indices.first >= _width || indices.second >= _height)
        {
            throw std::out_of_range("Index out of bounds!");
        }
        long pos = indices.second * _width + indices.first;

#ifdef DRIVELESS_CUDA_ENABLED
        return _frame->get()[pos];
#else
        return _frame.get()[pos];
#endif
    }

public:
    Frame(int width, int height, int numCPUThreadHandlers = 12);

    virtual void copyFrom(float *ptr);
    virtual void clear();

    inline T *getPtr()
    {
#ifdef DRIVELESS_CUDA_ENABLED
        return _frame->get();
#else
        return _frame.get();
#endif
    }

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

#ifdef DRIVELESS_CUDA_ENABLED
        return _frame[pos];
#else
        return _frame.get()[pos];
#endif
    }
};

template <typename T>
Frame<T>::Frame(int width, int height, int numCPUThreadHandlers) : _width(width), _height(height), _numCPUThreadHandlers(numCPUThreadHandlers)
{
    size_t size = _width * _height;
#ifdef DRIVELESS_CUDA_ENABLED
    this->_frame = std::make_unique<CudaPtr<T>>(size);
#else
    this->_frame = std::make_unique<T[]>(size);
#endif
}

template <typename T>
void Frame<T>::clear()
{
#ifdef DRIVELESS_CUDA_ENABLED
    CUDA_clear(_frame->get(), _width, _height);
#else
    ParallelClear<T>(_numCPUThreadHandlers, _frame.get(), _width, _height).runAndWait();
#endif
}

template <typename T>
void Frame<T>::copyFrom(float *ptr)
{
#ifdef DRIVELESS_CUDA_ENABLED
    ParallelCopy<T>(_numCPUThreadHandlers, _frame->get(), ptr, (int)_width, (int)_height).runAndWait();
#else
    ParallelCopy<T>(_numCPUThreadHandlers, _frame.get(), ptr, (int)_width, (int)_height).runAndWait();
#endif
}

#endif
