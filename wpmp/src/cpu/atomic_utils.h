#ifndef __ATOMIC_UTILS_WPMP_H
#define __ATOMIC_UTILS_WPMP_H

#ifndef DRIVELESS_CUDA_ENABLED
#include <atomic>

inline long long atomicMin(std::atomic<long long>& val, long long new_val) {
    long long old = val.load(std::memory_order_relaxed);
    while (new_val < old &&
           !val.compare_exchange_weak(old, new_val, std::memory_order_relaxed))
    {}
    return old;
}
#endif

#endif
