#pragma once

#ifndef __CPU_FRAME_DRIVELESS_H
#define __CPU_FRAME_DRIVELESS_H

#include <stdexcept>
#include "cpu_parallel_processor.h"
#include "cuda_basic.h"

#if defined(CUDA_VERSION_MAJOR) && CUDA_VERSION_MAJOR >= 13
#define double4 double4_16a
#endif

extern void copy_data(float *ptr, float4 *dest, long pos);
extern void copy_data(float *ptr, double4 *dest, long pos);
extern void copy_data(float *ptr, int4 *dest, long pos);
extern void copy_data(float *ptr, float3 *dest, long pos);
extern void copy_data(float *ptr, double3 *dest, long pos);
extern void copy_data(float *ptr, int3 *dest, long pos);
extern void copy_data(float *ptr, float2 *dest, long pos);
extern void copy_data(float *ptr, double2 *dest, long pos);
extern void copy_data(float *ptr, int2 *dest, long pos);
extern void copy_data(float *ptr, float *dest, long pos);
extern void copy_data(float *ptr, double *dest, long pos);
extern void copy_data(float *ptr, int *dest, long pos);

// extern void parallel_clear(double4 *_frame, int width, int height, int numThreadHandlers);
// extern void parallel_clear(float4 *_frame, int width, int height, int numThreadHandlers);
// extern void parallel_clear(int4 *_frame, int width, int height, int numThreadHandlers);
// extern void parallel_clear(double3 *_frame, int width, int height, int numThreadHandlers);
// extern void parallel_clear(float3 *_frame, int width, int height, int numThreadHandlers);
// extern void parallel_clear(int3 *_frame, int width, int height, int numThreadHandlers);
// extern void parallel_clear(double2 *_frame, int width, int height, int numThreadHandlers);
// extern void parallel_clear(float2 *_frame, int width, int height, int numThreadHandlers);
// extern void parallel_clear(int2 *_frame, int width, int height, int numThreadHandlers);
// extern void parallel_clear(double *_frame, int width, int height, int numThreadHandlers);
// extern void parallel_clear(float *_frame, int width, int height, int numThreadHandlers);
// extern void parallel_clear(int *_frame, int width, int height, int numThreadHandlers);

template <typename T>
class CPUframe
{
private:
    std::unique_ptr<T[]> _frame;
    const int _width;
    const int _height;
    int _numThreadHandlers;

protected:
    class ParallelCopy : public ParallelProcessor
    {
        T *_data;
        float *_ptr;
        int maxPos;

    public:
        ParallelCopy(int numThreadHandlers, T *frame, float *ptr, int width, int height)
            : ParallelProcessor(numThreadHandlers, width, width)
        {
            this->_data = frame;
            this->_ptr = ptr;
            this->maxPos = width * height;
        }

        void handler(int threadId) override
        {
            if (threadId >= maxPos)
                return;
            copy_data(_ptr, _data, threadId);
        }
    };

    class ParallelClear : public ParallelProcessor
    {
        T *_data;
        int maxPos;

    public:
        ParallelClear(int numThreadHandlers, T *frame, int width, int height)
            : ParallelProcessor(numThreadHandlers, width, width)
        {
            this->_data = frame;
            this->maxPos = width * height;
        }

        void handler(int threadId) override
        {
            if (threadId >= maxPos)
                return;
            _data[threadId] = static_cast<T>(0);
        }
    };

    T &at(std::pair<size_t, size_t> indices)
    {
        if (indices.first >= _width || indices.second >= _height)
        {
            throw std::out_of_range("Index out of bounds!");
        }
        long pos = indices.second * _width + indices.first;
        return _frame.get()[pos];
    }

public:
    CPUframe(int width, int height, int numThreadHandlers = 32);

    virtual void copyFrom(float *ptr);
    virtual void clear();
    inline T *getPtr() { return _frame.get(); }

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
        return _frame[pos];
    }
};

template <typename T>
CPUframe<T>::CPUframe(int width, int height, int numThreadHandlers) : _width(width), _height(height), _numThreadHandlers(numThreadHandlers)
{
    size_t size = _width * _height;
    this->_frame = std::make_unique<T[]>(size);
}

template <typename T>
void CPUframe<T>::copyFrom(float *ptr)
{
    CPUframe<T>::ParallelCopy(this->_numThreadHandlers, _frame.get(), ptr, _width, _height).runAndWait();
}

template <typename T>
void CPUframe<T>::clear()
{
    //parallel_clear(_frame.get(), _width, _height, _numThreadHandlers);
    CPUframe<T>::ParallelClear(_numThreadHandlers, _frame.get(), _width, _height).runAndWait();
}

#endif
