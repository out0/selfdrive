#pragma once

#ifndef __CPU_PARALLEL_CLEAR_DRIVELESS_H
#define __CPU_PARALLEL_CLEAR_DRIVELESS_H

#include "cuda_basic.h"
#include "cpu_parallel_processor.h"
#include <stdexcept>


template <typename T>
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

#endif
