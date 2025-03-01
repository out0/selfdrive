
#include "../include/cuda_utils.h"

#ifndef USE_JETSON_UTILS
cudaError_t cudaCheckError(cudaError_t retval, const char *txt, const char *file, int line)
{
    if (retval == cudaSuccess)
    {
#if !defined(CUDA_TRACE)
        return cudaSuccess;
#else
        LogDebug("[CUDA] %s\n", txt);
#endif
    }
    else
    {
        printf("[CUDA] %s\n", txt);
        printf("[CUDA] %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
        printf("[CUDA] %s:%i\n", file, line);
    }

    return retval;
}

cudaError_t cudaMallocMapped(void **ptr, size_t size, bool clear)
{
    void *cpu = NULL;
    void *gpu = NULL;

    if (!ptr || size == 0)
        return cudaErrorInvalidValue;

    // CUDA_ASSERT(cudaSetDeviceFlags(cudaDeviceMapHost));

    CUDA_ASSERT(cudaHostAlloc(&cpu, size, cudaHostAllocMapped));
    CUDA_ASSERT(cudaHostGetDevicePointer(&gpu, cpu, 0));

    if (cpu != gpu)
    {
        printf("[CUDA] cudaMallocMapped() - addresses of CPU and GPU pointers don't match (CPU=%p GPU=%p)\n", cpu, gpu);
        return cudaErrorInvalidDevicePointer;
    }

    if (clear)
        memset(cpu, 0, size);

    *ptr = cpu;
    return cudaSuccess;
}

// /**
//  * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
//  *
//  * @note this overload of cudaAllocMapped returns one pointer, assumes that the
//  *       CPU and GPU addresses will match (as is the case with any recent CUDA version).
//  *
//  * @param[out] ptr Returned pointer to the shared CPU/GPU memory.
//  * @param[in] size Size (in bytes) of the shared memory to allocate.
//  * @param[in] clear If `true` (default), the memory contents will be filled with zeros.
//  *
//  * @returns `true` if the allocation succeeded, `false` otherwise.
//  * @ingroup cudaMemory
//  */
// bool cudaAllocMapped(void **ptr, size_t size, bool clear)
// {
//     return CUDA_SUCCESS(cudaMallocMapped(ptr, size, clear));
// }



/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 *
 * @note although two pointers are returned, one for CPU and GPU, they both resolve to the same physical memory.
 *
 * @param[out] cpuPtr Returned CPU pointer to the shared memory.
 * @param[out] gpuPtr Returned GPU pointer to the shared memory.
 * @param[in] size Size (in bytes) of the shared memory to allocate.
 * @param[in] clear If `true` (the default), the memory contents will be filled with zeros.
 *
 * @returns `true` if the allocation succeeded, `false` otherwise.
 * @ingroup cudaMemory
 */
// bool cudaAllocMapped(void **cpuPtr, void **gpuPtr, size_t size, bool clear)
// {
//     if (!cpuPtr || !gpuPtr || size == 0)
//         return false;

//     // CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

//     if (CUDA_FAILED(cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped)))
//         return false;

//     if (CUDA_FAILED(cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0)))
//         return false;

//     if (clear)
//         memset(*cpuPtr, 0, size);

//     printf("[CUDA] cudaAllocMapped %zu bytes, CPU %p GPU %p\n", size, *cpuPtr, *gpuPtr);
//     return true;
// }

#endif
