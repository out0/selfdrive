#pragma once

#ifndef __CPU_PARALLEL_COPY_DRIVELESS_H
#define __CPU_PARALLEL_COPY_DRIVELESS_H

#include "cuda_basic.h"
#include "cpu_parallel_processor.h"
#include <stdexcept>

extern void copy_data(float *ptr, float4 *dest, long pos);
extern void copy_data(float *ptr, DOUBLE4 *dest, long pos);
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


template <typename T>
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

#endif
