#ifndef H_CUDA_BASIC
#define H_CUDA_BASIC

#define HEADING_FROM_START 0x10
#define HEADING_0 0x08
#define HEADING_90 0x04
#define HEADING_45 0x02
#define HEADING_MINUS_45 0x01


#include <cmath>
#include <cstring>

// #define MINIMAL_DISTANCE_X 3
// #define MINIMAL_DISTANCE_Z 2

#ifdef USE_JETSON_UTILS
#include <jetson-utils/cudaMappedMemory.h>

if (!cudaAllocMapped(&params, sizeof(int) * 6))
{
}

#else
#include <cuda_runtime.h>
#include <stdio.h>

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


#endif
#endif