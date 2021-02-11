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

//void
//copyToSymbol(const void* symbol, const void* src, size_t count) {
//    cudaError_t err = cudaMemcpyToSymbol(symbol, src, count, 0, cudaMemcpyHostToDevice);
//    if (err != cudaSuccess)
//    {
//        fprintf(stderr, "Failed to copy a constant symbol to the device (error code %s)!\n", cudaGetErrorString(err));
//        exit(EXIT_FAILURE);
//    }
//}

void
copyToDevice(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy a buffer from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void
copyToHost(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy a buffer from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void
swapBuffers(T** a, T** b) {
    T* t = *a;
    *a = *b;
    *b = t;
}

void
cudaUnalloc(void *ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device buffer (error code %s)!\n", cudaGetErrorString(err));
    }
}


template <typename T>
class SingleBuffer {
public:
    T* h_Current = NULL;
    T* d_Current = NULL;
    size_t count;
    size_t size;

    SingleBuffer(size_t count)
        : count(count), size(count * sizeof(T))
    {
        universalAlloc(&h_Current, &d_Current, count);
        memset(h_Current, 0, size);
    }

    void
    copyToDevice() {
        ::copyToDevice(d_Current, h_Current, size);
    }

    void
    copyToHost() {
        ::copyToHost(h_Current, d_Current, size);
    }

    void
    clearOnDevice() {
        cudaMemset(d_Current, 0, size);
    }

    ~SingleBuffer() {
        cudaUnalloc(d_Current);
        free(h_Current);
    }
};

template <typename T>
class DoubleBuffer : public SingleBuffer<T> {
public:
    T* d_Next = NULL;

    DoubleBuffer(size_t count)
        : SingleBuffer(count)
    {
        cudaAlloc(&d_Next, count);
    }

    void
    swap() {
        swapBuffers(&d_Current, &d_Next);
    }

    void
    clearNextOnDevice() {
        cudaMemset(d_Next, 0, size);
    }

    ~DoubleBuffer() {
        cudaUnalloc(d_Next);
    }
};

template <typename T>
class DeviceOnlySingleBuffer {
public:
    T* d_Current = NULL;
    size_t count;
    size_t size;

    DeviceOnlySingleBuffer(size_t count)
        : count(count), size(count * sizeof(T))
    {
        cudaAlloc(&d_Current, count);
    }

    void
    clear() {
        cudaMemset(d_Current, 0, size);
    }

    ~DeviceOnlySingleBuffer() {
        cudaUnalloc(d_Current);
    }
};

template <typename T>
class DeviceOnlyDoubleBuffer : public DeviceOnlySingleBuffer<T> {
public:
    T* d_Next = NULL;

    DeviceOnlyDoubleBuffer(size_t count)
        : DeviceOnlySingleBuffer(count)
    {
        cudaAlloc(&d_Next, count);
    }

    void
    swap() {
        swapBuffers(&d_Current, &d_Next);
    }

    ~DeviceOnlyDoubleBuffer() {
        cudaUnalloc(d_Next);
    }
};
