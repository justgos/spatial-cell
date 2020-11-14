/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <ctime>
#include <ratio>
#include <chrono>
#include <random>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>
//#include <crt/math_functions.h>
#include <cub/cub.cuh>
//#include <cub/device/device_radix_sort.cuh>

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

#define SIM_SIZE 0.1f
// Grid size on each side will be 2^N_GRID_CELLS_BITS
#define N_GRID_CELLS_BITS 4
#define N_GRID_CELLS (1 << N_GRID_CELLS_BITS)
#define GRID_CELL_SIZE (SIM_SIZE / N_GRID_CELLS)
#define GRID_SIZE (N_GRID_CELLS * N_GRID_CELLS * N_GRID_CELLS)

struct Particle {
    float3 pos;
    float3 velocity;
    int type;
};

__device__ __inline__ unsigned int
getGridIdx(float coord)
{
    return (unsigned int)(coord / GRID_CELL_SIZE);
}

__device__ __inline__ unsigned int
makeIdx(unsigned int gridX, unsigned int gridY, unsigned int gridZ)
{
    return (gridX << (2 * N_GRID_CELLS_BITS))
        | (gridY << N_GRID_CELLS_BITS)
        | (gridZ << 0);
}

__device__ __inline__ unsigned int
makeIdx(Particle p)
{
    return makeIdx(getGridIdx(p.pos.x), getGridIdx(p.pos.y), getGridIdx(p.pos.z));
}

__global__ void
setupKernel(curandState* rngState, int numElements) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= numElements)
        return;

    /*curand_init(42, idx, 0, &rngState[idx]);*/

    // Faster initialization
    // Ref: https://forums.developer.nvidia.com/t/curand-initialization-time/19758/3
    curand_init((42 << 24) + idx, 0, 0, &rngState[idx]);
}

__global__ void
updateIndices(const Particle* curParticles, unsigned int* indices, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numElements)
        return;

    indices[idx] = makeIdx(curParticles[idx]);
}

__global__ void
updateGridRanges(const unsigned int* indices, unsigned int* gridRanges, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= GRID_SIZE)
        return;

    const int startIndex = idx * (long long)numElements / GRID_SIZE;
    const int endIndex = min((idx + 1) * (long long)numElements / GRID_SIZE, (long long)numElements);

    int lastIdx = indices[startIndex];
    if (startIndex <= 0 || indices[startIndex - 1] != lastIdx)
        gridRanges[lastIdx * 2] = startIndex;

    for (int i = startIndex; i < endIndex; i++) {
        int curIdx = indices[i];
        if (curIdx != lastIdx) {
            gridRanges[lastIdx * 2 + 1] = i;
            gridRanges[curIdx * 2] = i;
            lastIdx = curIdx;
        }
    }
    if(endIndex >= numElements || indices[endIndex] != lastIdx)
        gridRanges[lastIdx * 2 + 1] = endIndex;
}

__global__ void
move(curandState* rngState, const Particle *curParticles, Particle *nextParticles, unsigned int* indices, unsigned int* gridRanges, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numElements)
        return;

    Particle p = curParticles[idx];

    float maxDist = 0.007;

    float3 moveVec = make_float3(0.0f, 0.0f, 0.0f);
    float3 attractionVec = make_float3(0.0f, 0.0f, 0.0f);
    constexpr float clipImpulse = 10.0f;
    constexpr float impulseScale = 0.0001f;
    //constexpr float distScale = 100.0f;

    const int cgx = getGridIdx(p.pos.x),
        cgy = getGridIdx(p.pos.y),
        cgz = getGridIdx(p.pos.z);

    float nPartners = 0.0;
    for (int gx = max(cgx - 1, 0); gx <= min(cgx + 1, N_GRID_CELLS - 1); gx++) {
        for (int gy = max(cgy - 1, 0); gy <= min(cgy + 1, N_GRID_CELLS - 1); gy++) {
            for (int gz = max(cgz - 1, 0); gz <= min(cgz + 1, N_GRID_CELLS - 1); gz++) {
                /*if (gx == cgx && gy == cgy && gz == cgz)
                    continue;*/
                const unsigned int startIdx = gridRanges[makeIdx(gx, gy, gz) * 2];
                const unsigned int endIdx = gridRanges[makeIdx(gx, gy, gz) * 2 + 1];
                for (int j = startIdx; j < endIdx; j++) {
                    if (j == idx)
                        continue;
                    const Particle tp = curParticles[j];
                    float3 delta = make_float3(
                        tp.pos.x - p.pos.x,
                        tp.pos.y - p.pos.y,
                        tp.pos.z - p.pos.z
                    );
                    if (fabs(delta.x) > maxDist
                        || fabs(delta.y) > maxDist
                        || fabs(delta.y) > maxDist)
                        continue;

                    float dist = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
                    float repulsion = -fmin(1 / (pow(dist * 400.0f, 2.0f) + 1e-6f), clipImpulse) * impulseScale;
                    float attraction = 0.0f;
                    // Graph: https://www.desmos.com/calculator/wdnrfaaqps
                    if (p.type == 0 && p.type == tp.type) {
                        attraction = 0.7 * (exp(-pow(abs(dist) * 100.0f, 2.0f)) * impulseScale * 10 - fmin(1 / (pow(dist * 70.0f, 2.0f) + 1e-6f), clipImpulse) * impulseScale);
                        attractionVec.x += copysign(1.0, delta.x) * (attraction);
                        attractionVec.y += copysign(1.0, delta.y) * (attraction);
                        attractionVec.z += copysign(1.0, delta.z) * (attraction);
                    }
                    moveVec.x += copysign(1.0, delta.x) * (repulsion);
                    moveVec.y += copysign(1.0, delta.y) * (repulsion);
                    moveVec.z += copysign(1.0, delta.z) * (repulsion);
                    nPartners += 1.0;
                }
            }
        }
    }

    // Prevent attraction overkill for large aggregations
    nPartners = pow(fmax(nPartners, 1.0f), 0.5f);
    attractionVec.x /= nPartners;
    attractionVec.y /= nPartners;
    attractionVec.z /= nPartners;
    moveVec.x += attractionVec.x;
    moveVec.y += attractionVec.y;
    moveVec.z += attractionVec.z;

    float movementNoiseScale = 0.0001f;
    moveVec.x += (curand_normal(&rngState[idx]) - 0.0) * movementNoiseScale;
    moveVec.y += (curand_normal(&rngState[idx]) - 0.0) * movementNoiseScale;
    moveVec.z += (curand_normal(&rngState[idx]) - 0.0) * movementNoiseScale;

    const float velocityDecay = 0.7;
    p.velocity.x *= velocityDecay;
    p.velocity.y *= velocityDecay;
    p.velocity.z *= velocityDecay;
    p.velocity.x += moveVec.x;
    p.velocity.y += moveVec.y;
    p.velocity.z += moveVec.z;

    p.pos.x = fmin(fmax(p.pos.x + p.velocity.x, 0.0f), SIM_SIZE);
    p.pos.y = fmin(fmax(p.pos.y + p.velocity.y, 0.0f), SIM_SIZE);
    p.pos.z = fmin(fmax(p.pos.z + p.velocity.z, 0.0f), SIM_SIZE);
    /*p.pos.x += moveVec.x;
    p.pos.y += moveVec.y;
    p.pos.z += moveVec.z;*/
    nextParticles[idx] = p;
}

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

void
printCUDAIntArray(unsigned int* a, unsigned int len) {
    unsigned int* host_a = new unsigned int[len];
    cudaMemcpy(host_a, a, len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("[");
    for (unsigned int i = 0; i < len; i++) {
        printf(" %d", host_a[i]);
    }
    printf(" ]\n");
    free(host_a);
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    std::mt19937 rngGen(42);
    std::uniform_real_distribution<double> rng(0.0, 1.0);

    int numParticles = 1 * 2 * 1024;
    size_t count = numParticles;
    size_t size = count * sizeof(Particle);
    printf("[Simulating %d particles]\n", numParticles);

    Particle *h_Particles, *h_NextParticles;
    Particle *d_Particles = NULL,
        *d_NextParticles = NULL;
    universalAlloc(&h_Particles, &d_Particles, count);
    universalAlloc(&h_NextParticles, &d_NextParticles, count);
    unsigned int *d_Indices = NULL,
        *d_NextIndices = NULL,
        *d_GridRanges;
    cudaAlloc(&d_Indices, count);
    cudaAlloc(&d_NextIndices, count);
    cudaAlloc(&d_GridRanges, GRID_SIZE * 2);

    curandState* rngState;
    cudaAlloc(&rngState, count);

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_Indices, d_NextIndices, d_Particles, d_NextParticles, count);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    printf("temp_storage_bytes %d\n", temp_storage_bytes);

    int numTypes = 3;

    // Initialize the host input vectors
    for (int i = 0; i < numParticles; i++)
    {
        h_Particles[i].pos = make_float3(
            rng(rngGen) * SIM_SIZE,
            rng(rngGen) * SIM_SIZE,
            rng(rngGen) * SIM_SIZE
        );
        h_Particles[i].velocity = make_float3(0, 0, 0);
        h_Particles[i].type = rng(rngGen) * numTypes;
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_Particles, h_Particles, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int particleBlocksPerGrid =(numParticles + threadsPerBlock - 1) / threadsPerBlock;
    printf("Particle CUDA kernel with %d blocks of %d threads\n", particleBlocksPerGrid, threadsPerBlock);

    int gridBlocksPerGrid = (GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    printf("Grid CUDA kernel with %d blocks of %d threads\n", gridBlocksPerGrid, threadsPerBlock);

    setupKernel KERNEL_ARGS2(particleBlocksPerGrid, threadsPerBlock) (rngState, numParticles);

    std::ofstream fout;
    fout.open("./results/frames.dat", std::ios::binary | std::ios::out);
    const float simSize = SIM_SIZE;
    fout.write((char*)&simSize, sizeof(float));
    // Particle buffer size
    fout.write((char*)&numParticles, sizeof(unsigned int));

    // Number of existing particles
    fout.write((char*)&numParticles, sizeof(unsigned int));
    fout.write((char*)h_Particles, size);
    
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

    constexpr int nIter = 101;
    for (int i = 0; i < nIter; i++) {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        updateIndices KERNEL_ARGS2(particleBlocksPerGrid, threadsPerBlock) (d_Particles, d_Indices, numParticles);
        cudaDeviceSynchronize();
        std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_Indices, d_NextIndices, d_Particles, d_NextParticles, count);
        cudaDeviceSynchronize();
        std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
        swapBuffers(&d_Particles, &d_NextParticles);
        swapBuffers(&d_Indices, &d_NextIndices);
        //printCUDAIntArray(d_Indices, numParticles);
        cudaMemset(d_GridRanges, 0, GRID_SIZE * 2 * sizeof(unsigned int));
        updateGridRanges KERNEL_ARGS2(gridBlocksPerGrid, threadsPerBlock) (d_Indices, d_GridRanges, numParticles);
        cudaDeviceSynchronize();
        std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
        //printCUDAIntArray(d_GridRanges, GRID_SIZE * 2);
        move KERNEL_ARGS2(particleBlocksPerGrid, threadsPerBlock) (rngState, d_Particles, d_NextParticles, d_Indices, d_GridRanges, numParticles);
        swapBuffers(&d_Particles, &d_NextParticles);
        cudaDeviceSynchronize();
        std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();

        if (i % 10 == 0) {
            printf("updateIndices %f, SortPairs %f, updateGridRanges %f, move %f\n",
                (std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t1)).count(),
                (std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3)).count(),
                (std::chrono::duration_cast<std::chrono::duration<double>>(t5 - t4)).count(),
                (std::chrono::duration_cast<std::chrono::duration<double>>(t6 - t5)).count()
            );
        }

        cudaMemcpy(h_Particles, d_Particles, size, cudaMemcpyDeviceToHost);
        fout.write((char*)&numParticles, sizeof(unsigned int));
        fout.write((char*)h_Particles, size);
    }
    cudaDeviceSynchronize();
    fout.close();
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t0);
    printf("time %f\n", time_span.count());

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_Particles, d_Particles, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float minX = 0.5;
    for (int i = 0; i < numParticles; i++) {
        minX = fmin(minX, h_Particles[i].pos.x);
    }
    printf("minX, %f\n", minX);

    // Verify that the result vector is correct
    /*for (int i = 0; i < numParticles; ++i)
    {
        if (fabs(h_Pos[i] + h_B[i] - h_NextPos[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }*/

    // Free device global memory
    err = cudaFree(d_Particles);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_NextParticles);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_Indices);
    cudaFree(d_NextIndices);
    cudaFree(d_GridRanges);
    cudaFree(d_temp_storage);

    // Free host memory
    free(h_Particles);
    free(h_NextParticles);

    printf("Done\n");
    return 0;
}

