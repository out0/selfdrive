
#ifndef H_CUDA_UTILS
#define H_CUDA_UTILS

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

#define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)
#define CUDA_SUCCESS(x)		(CUDA(x) == cudaSuccess) 
#define CUDA_FAILED(x)		(CUDA(x) != cudaSuccess)
#define CUDA_VERIFY(x)		if(CUDA_FAILED(x))	return false;
#define CUDA_ASSERT(x)      { const cudaError_t _retval = CUDA(x); if(_retval != cudaSuccess) return _retval; }
#define CUDA_FREE(x) 		if(x != NULL) { cudaFree(x); x = NULL; }
#define CUDA_FREE_HOST(x)	if(x != NULL) { cudaFreeHost(x); x = NULL; }
#define SAFE_DELETE(x) 		if(x != NULL) { delete x; x = NULL; }
#define SAFE_FREE(x) 		if(x != NULL) { free(x); x = NULL; }

cudaError_t cudaCheckError(cudaError_t retval, const char *txt, const char *file, int line);
cudaError_t cudaMallocMapped(void **ptr, size_t size, bool clear = true);
/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 *
 * @note this overload of cudaAllocMapped returns one pointer, assumes that the
 *       CPU and GPU addresses will match (as is the case with any recent CUDA version).
 *
 * @param[out] ptr Returned pointer to the shared CPU/GPU memory.
 * @param[in] size Size (in bytes) of the shared memory to allocate.
 * @param[in] clear If `true` (default), the memory contents will be filled with zeros.
 *
 * @returns `true` if the allocation succeeded, `false` otherwise.
 * @ingroup cudaMemory
 */
//bool cudaAllocMapped(void **ptr, size_t size, bool clear = true);
/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 *
 * This is a templated version for allocating images from vector types
 * like uchar3, uchar4, float3, float4, ect.  The overall size of the
 * allocation is specified by the size parameter.
 *
 * @param[out] ptr Returned pointer to the shared CPU/GPU memory.
 * @param[in] size size of the allocation, in bytes.
 * @param[in] clear If `true` (default), the memory contents will be filled with zeros.
 *
 * @returns `true` if the allocation succeeded, `false` otherwise.
 * @ingroup cudaMemory
 */
/**
 * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
 *
 * This is a templated version for allocating images from vector types
 * like uchar3, uchar4, float3, float4, ect.  The overall size of the
 * allocation is specified by the size parameter.
 *
 * @param[out] ptr Returned pointer to the shared CPU/GPU memory.
 * @param[in] size size of the allocation, in bytes.
 * @param[in] clear If `true` (default), the memory contents will be filled with zeros.
 *
 * @returns `true` if the allocation succeeded, `false` otherwise.
 * @ingroup cudaMemory
 */
template <typename T>
inline bool cudaAllocMapped(T **ptr, size_t size, bool clear = true)
{
    cudaError_t p = cudaMallocMapped((void **)ptr, size, clear);
    return CUDA_SUCCESS(p);
}

// /**
//  * Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
//  *
//  * @note although two pointers are returned, one for CPU and GPU, they both resolve to the same physical memory.
//  *
//  * @param[out] cpuPtr Returned CPU pointer to the shared memory.
//  * @param[out] gpuPtr Returned GPU pointer to the shared memory.
//  * @param[in] size Size (in bytes) of the shared memory to allocate.
//  * @param[in] clear If `true` (the default), the memory contents will be filled with zeros.
//  *
//  * @returns `true` if the allocation succeeded, `false` otherwise.
//  * @ingroup cudaMemory
//  */
// bool cudaAllocMapped(void **cpuPtr, void **gpuPtr, size_t size, bool clear = true);

#endif
#endif