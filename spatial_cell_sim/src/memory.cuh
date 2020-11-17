#pragma once

#include <cuda_runtime.h>


template <typename T>
void
cudaAlloc(T** deviceVar, size_t count) {
    size_t size = count * sizeof(T);
    cudaError_t err = cudaMalloc((void**)deviceVar, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate a device buffer (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void
cudaAlloc(T** deviceVar, size_t count, T *initialValue) {
    cudaAlloc(deviceVar, count);
    size_t size = count * sizeof(T);
    cudaError_t err = cudaMemcpy(*deviceVar, initialValue, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set the initial value for a device buffer (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void
universalAlloc(T** hostVar, T** deviceVar, size_t count) {
    size_t size = count * sizeof(T);
    *hostVar = (T*)malloc(size);

    if (hostVar == NULL)
    {
        fprintf(stderr, "Failed to allocate a host buffer!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t err = cudaMalloc((void**)deviceVar, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate a device buffer (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void
swapBuffers(T* a, T* b) {
    T t = *a;
    *a = *b;
    *b = t;
}
