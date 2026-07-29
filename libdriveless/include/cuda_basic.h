#ifndef __CUDA_BASIC_DRIVELESS_H
#define __CUDA_BASIC_DRIVELESS_H


#include <stdio.h>
#include <memory>
#include <vector>
#include <cstring>

#include "driveless_config.h"

#ifdef DRIVELESS_CUDA_ENABLED
#include <cuda_runtime.h>
#include <math_constants.h>
#else
#define CUDART_PI_F             3.141592654F
#endif

#define THREADS_IN_BLOCK 256
#define BIT_HEADING_90 0x80
#define BIT_HEADING_67_5 0x40
#define BIT_HEADING_45 0x20
#define BIT_HEADING_22_5 0x10
#define BIT_HEADING_0 0x08
#define BIT_HEADING_MINUS_22_5 0x04
#define BIT_HEADING_MINUS_45 0x02
#define BIT_HEADING_MINUS_67_5 0x01

#define ANGLE_HEADING_0 0.0
#define ANGLE_HEADING_22_5 CUDART_PI_F / 8
#define ANGLE_HEADING_45 CUDART_PI_F / 4
#define ANGLE_HEADING_67_5 (3*CUDART_PI_F) / 8
#define ANGLE_HEADING_90 CUDART_PI_F / 2
#define ANGLE_HEADING_MINUS_22_5 -CUDART_PI_F / 8
#define ANGLE_HEADING_MINUS_45 -CUDART_PI_F / 4
#define ANGLE_HEADING_MINUS_67_5 -(3*CUDART_PI_F) / 8

#define TOP 8       // 1000
#define BOTTOM 4    // 0100
#define LEFT 2      // 0010
#define RIGHT  1    // 0001
#define INSIDE 0    // 0000 


// #define MINIMAL_DISTANCE_X 3
// #define MINIMAL_DISTANCE_Z 2

#ifdef USE_JETSON_UTILS
#include <jetson-utils/cudaMappedMemory.h>

if (!cudaAllocMapped(&params, sizeof(int) * 6))
{
}

#else

#ifdef DRIVELESS_CUDA_ENABLED


#if defined(CUDA_VERSION_MAJOR) && CUDA_VERSION_MAJOR >= 13
using DOUBLE4=double4_16a;
#else
using DOUBLE4=double4;
#endif

#define CUDA(x) cudaCheckError((x), #x, __FILE__, __LINE__)
#define CUDA_SUCCESS(x) (CUDA(x) == cudaSuccess)
#define CUDA_FAILED(x) (CUDA(x) != cudaSuccess)
#define CUDA_VERIFY(x)  \
    if (CUDA_FAILED(x)) \
        return false;

inline cudaError_t cudaCheckError(cudaError_t retval, const char *txt, const char *file, int line)
{
#if !defined(CUDA_TRACE)
    if (retval == cudaSuccess)
        return cudaSuccess;
#endif

    // int activeDevice = -1;
    // cudaGetDevice(&activeDevice);

    // Log("[cuda]   device %i  -  %s\n", activeDevice, txt);

    if (retval == cudaSuccess)
    {
        printf("[cuda] %s\n", txt);
    }
    else
    {
        printf("[cuda] %s\n", txt);
    }

    if (retval != cudaSuccess)
    {
        printf("[cuda]  %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
        printf("[cuda]  %s:%i\n", file, line);
    }

    return retval;
}

inline bool cudaAllocMapped(void **cpuPtr, void **gpuPtr, size_t size)
{
    if (!cpuPtr || !gpuPtr || size == 0)
        return false;

    // CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

    if (CUDA_FAILED(cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped)))
        return false;

    if (CUDA_FAILED(cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0)))
        return false;

    memset(*cpuPtr, 0, size);
    // LogDebug(LOG_CUDA "cudaAllocMapped %zu bytes, CPU %p GPU %p\n", size, *cpuPtr, *gpuPtr);
    return true;
}

inline bool cudaAllocMapped(void **ptr, size_t size)
{
    void *cpuPtr = NULL;
    void *gpuPtr = NULL;

    if (!ptr || size == 0)
        return false;

    if (!cudaAllocMapped(&cpuPtr, &gpuPtr, size))
        return false;

    if (cpuPtr != gpuPtr)
    {
        printf("[cuda] cudaAllocMapped() - addresses of CPU and GPU pointers don't match\n");
        return false;
    }

    *ptr = gpuPtr;
    return true;
}

template <typename T>
inline bool cudaAllocMapped(T **ptr, size_t size)
{
    return cudaAllocMapped((void **)ptr, size);
}


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


inline cptr<float4> copyToCudaMemory(float *path, int count)
{
    cptr<float4> data = std::make_unique<CudaPtr<float4>>(count);
    float4 *addr = data->get();

    for (int c = 0; c < count; c++)
    {
        int pos = 4 * c;
        addr[c].x = path[pos];
        addr[c].y = path[pos + 1];
        addr[c].z = path[pos + 2];
        addr[c].w = 0.0;
    }
    return data;
}

#else
#ifndef __device__ 
#define __device__ 
#endif
#ifndef __host__ 
#define __host__ 
#endif

typedef struct uchar2 {
    unsigned char x;
    unsigned char y;
} uchar2;
typedef struct uchar3 {
    unsigned char x;
    unsigned char y;
    unsigned char z;
} uchar3;
typedef struct uchar4 {
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
} uchar4;

typedef struct int2 {
    int x;
    int y;
} int2;

typedef struct uint2 {
    unsigned int x;
    unsigned int y;
} uint2;

typedef struct float2 {
    float x;
    float y;
} float2;
typedef struct double2 {
    double x;
    double y;
} double2;
typedef struct int3 {
    int x;
    int y;
    int z;
} int3;
typedef struct float3 {
    float x;
    float y;
    float z;
} float3;
typedef struct double3 {
    double x;
    double y;
    double z;
} double3;
typedef struct int4 {
    int x;
    int y;
    int z;
    int w;
} int4;
typedef struct float4 {
    float x;
    float y;
    float z;
    float w;
} float4;
typedef struct double4 {
    double x;
    double y;
    double z;
    double w;
} double4;

using DOUBLE4=double4;

#endif

__device__ __host__ inline int COMPUTE_POS(int width, int x, int z)
{
    return z * width + x;
}

// CODE:END

#endif

#include "waypoint.h"
std::unique_ptr<float4[]> copyToCpuMemory(std::vector<Waypoint> points);
std::unique_ptr<float4[]> copyToCpuMemory(float *path, int count);
#endif


