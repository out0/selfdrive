#pragma once

#ifndef __CUDA_PTR_DRIVELESS_H
#define __CUDA_PTR_DRIVELESS_H
#include <memory>
#include "cuda_basic.h"


template <typename T> 
class CudaPtr {
    T* _data;
    bool _data_owner;
    unsigned int _count;

public:

    CudaPtr() { 
        _data = nullptr;
        _data_owner = false;
        _count = 0;
    }

    CudaPtr(unsigned int count) {
        if (!cudaAllocMapped(&_data, sizeof(T)*count))
            throw std::bad_alloc();
        _data_owner = true;
        _count = count;
    }

    CudaPtr(T *val, int count) {
        _data = val;
        _data_owner = true;
        _count = count;
    }

    ~CudaPtr() {
        if (!_data_owner || _data == nullptr)
            return;
        cudaFreeHost(_data);
    }

    T* get () {
        return _data;
    }

    unsigned int count() {
        return _count;
    }

};

template <typename T>
using cptr = std::unique_ptr<CudaPtr<T>>; 
template <typename T>
using sptr = std::shared_ptr<CudaPtr<T>>; 

#endif