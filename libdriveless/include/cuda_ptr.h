#pragma once

#ifndef __CUDA_PTR_DRIVELESS_H
#define __CUDA_PTR_DRIVELESS_H
#include <memory>
#include "cuda_basic.h"


template <typename T> 
class CudaPtr {
    T* _data;
    bool _data_owner;

public:

    CudaPtr() { 
        _data = nullptr;
        _data_owner = false;
    }

    CudaPtr(unsigned int count) {
        if (!cudaAllocMapped(&_data, sizeof(T)*count))
            throw std::bad_alloc();
        _data_owner = true;
    }

    CudaPtr(T *val) {
        _data = val;
        _data_owner = true;
    }

    ~CudaPtr() {
        if (!_data_owner || _data == nullptr)
            return;
        cudaFreeHost(_data);
    }

    T* get () {
        return _data;
    }

};

template <typename T>
using cptr = std::unique_ptr<CudaPtr<T>>; 
template <typename T>
using sptr = std::shared_ptr<CudaPtr<T>>; 

#endif