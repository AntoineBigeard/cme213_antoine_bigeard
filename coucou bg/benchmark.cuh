#ifndef _BENCHMARK_CUH
#define _BENCHMARK_CUH

#include "util.cuh"

// Kernel for the benchmark
__global__ void elementwise_add(const int *x, const int *y,
                                int *z, unsigned int stride,
                                unsigned int size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    for (int i = index; i < size; i++) {
        z[i * stride] = x[i * stride] + y[i * stride];
    }
}

#endif
