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
#include <crt/math_functions.h>
#include <cub/cub.cuh>
//#include <cub/device/device_radix_sort.cuh>

#include <json/json.h>

#include "types.cuh"
#include "constants.cuh"
#include "macros.cuh"
#include "math.cuh"
#include "grid.cuh"
#include "memory.cuh"


__global__ void
setupKernel(curandState* rngState) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= d_Config.numParticles)
        return;

    /*curand_init(42, idx, 0, &rngState[idx]);*/

    // Faster initialization
    // Ref: https://forums.developer.nvidia.com/t/curand-initialization-time/19758/3
    curand_init((42 << 24) + idx, 0, 0, &rngState[idx]);
}

__global__ void
move(
    curandState* rngState,
    const Particle *curParticles,
    Particle *nextParticles,
    int *nActiveParticles,
    int *lastActiveParticle,
    unsigned int* indices,
    unsigned int* gridRanges
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= d_Config.numParticles)
        return;

    Particle p = curParticles[idx];

    if (!(p.flags & PARTICLE_FLAG_ACTIVE)) {
        nextParticles[idx] = p;
        return;
    }

    float3 moveVec = make_float3(0.0f, 0.0f, 0.0f);
    float3 attractionVec = make_float3(0.0f, 0.0f, 0.0f);
    constexpr float clipImpulse = 10.0f;
    constexpr float impulseScale = 0.0001f;
    //constexpr float distScale = 100.0f;

    const int cgx = getGridIdx(p.pos.x),
        cgy = getGridIdx(p.pos.y),
        cgz = getGridIdx(p.pos.z);

    bool shouldBeRemoved = false;
    float nPartners = 0.0;
    for (int gx = max(cgx - 1, 0); gx <= min(cgx + 1, d_Config.nGridCells - 1); gx++) {
        for (int gy = max(cgy - 1, 0); gy <= min(cgy + 1, d_Config.nGridCells - 1); gy++) {
            for (int gz = max(cgz - 1, 0); gz <= min(cgz + 1, d_Config.nGridCells - 1); gz++) {
                /*if (gx == cgx && gy == cgy && gz == cgz)
                    continue;*/
                const unsigned int startIdx = gridRanges[makeIdx(gx, gy, gz) * 2];
                const unsigned int endIdx = gridRanges[makeIdx(gx, gy, gz) * 2 + 1];
                for (int j = startIdx; j < endIdx; j++) {
                    if (j == idx)
                        continue;
                    const Particle tp = curParticles[j];
                    if (!(tp.flags & PARTICLE_FLAG_ACTIVE))
                        continue;
                    float3 delta = make_float3(
                        tp.pos.x - p.pos.x,
                        tp.pos.y - p.pos.y,
                        tp.pos.z - p.pos.z
                    );
                    if (fabs(delta.x) > d_Config.interactionDistance
                        || fabs(delta.y) > d_Config.interactionDistance
                        || fabs(delta.y) > d_Config.interactionDistance)
                        continue;

                    float dist = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);

                    if (p.type == 0 && tp.type == 1 && dist <= 0.007) {
                        shouldBeRemoved = true;
                    }
                    if (p.type == 1 && tp.type == 1 && dist <= 0.007) {
                        int newIdx = atomicAdd(lastActiveParticle, 1);
                        if (newIdx < d_Config.numParticles) {
                            Particle np = Particle();
                            np.pos = make_float3(
                                p.pos.x + delta.x / 2,
                                p.pos.y + delta.y / 2,
                                p.pos.z + delta.z / 2
                            );
                            np.velocity = make_float3(0, 0, 0);
                            np.type = 1;
                            np.flags = PARTICLE_FLAG_ACTIVE;
                            nextParticles[newIdx] = np;
                            atomicAdd(nActiveParticles, 1);
                        }
                    }

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

    if (shouldBeRemoved) {
        p.flags = p.flags ^ PARTICLE_FLAG_ACTIVE;
        atomicAdd(nActiveParticles, -1);
    }

    // Prevent attraction overkill for large aggregations
    nPartners = pow(fmax(nPartners, 1.0f), 0.5f);
    attractionVec.x /= nPartners;
    attractionVec.y /= nPartners;
    attractionVec.z /= nPartners;
    moveVec.x += attractionVec.x;
    moveVec.y += attractionVec.y;
    moveVec.z += attractionVec.z;

    moveVec.x += (curand_normal(&rngState[idx]) - 0.0) * d_Config.movementNoiseScale;
    moveVec.y += (curand_normal(&rngState[idx]) - 0.0) * d_Config.movementNoiseScale;
    moveVec.z += (curand_normal(&rngState[idx]) - 0.0) * d_Config.movementNoiseScale;

    p.velocity.x *= d_Config.velocityDecay;
    p.velocity.y *= d_Config.velocityDecay;
    p.velocity.z *= d_Config.velocityDecay;
    p.rot = lepr(p.rot, random_rotation(&rngState[idx]), d_Config.rotationNoiseScale);
    p.velocity.x += moveVec.x;
    p.velocity.y += moveVec.y;
    p.velocity.z += moveVec.z;

    p.pos.x = fmin(fmax(p.pos.x + p.velocity.x, 0.0f), d_Config.simSize);
    p.pos.y = fmin(fmax(p.pos.y + p.velocity.y, 0.0f), d_Config.simSize);
    p.pos.z = fmin(fmax(p.pos.z + p.velocity.z, 0.0f), d_Config.simSize);
    /*p.pos.x += moveVec.x;
    p.pos.y += moveVec.y;
    p.pos.z += moveVec.z;*/
    nextParticles[idx] = p;
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
    std::ifstream configFile("./config.json");
    Json::Value configJson;
    configFile >> configJson;

    Config config;
    config.numParticles = configJson["numParticles"].asInt();
    config.steps = configJson["steps"].asInt();
    config.simSize = configJson["simSize"].asFloat();
    config.interactionDistance = configJson["interactionDistance"].asFloat();
    config.nGridCellsBits = configJson["nGridCellsBits"].asInt();
    config.nGridCells = 1 << config.nGridCellsBits;
    config.gridCellSize = config.simSize / config.nGridCells;
    config.gridSize = config.nGridCells * config.nGridCells * config.nGridCells;
    config.movementNoiseScale = configJson["movementNoiseScale"].asFloat();
    config.rotationNoiseScale = configJson["rotationNoiseScale"].asFloat();
    config.velocityDecay = configJson["velocityDecay"].asFloat();

    printf("[Config]\n");
    printf("numParticles %d\n", config.numParticles);
    printf("steps %d\n", config.steps);
    printf("simSize %f\n", config.simSize);
    printf("interactionDistance %f\n", config.interactionDistance);
    printf("nGridCellsBits %d\n", config.nGridCellsBits);
    printf("nGridCells %d\n", config.nGridCells);
    printf("gridCellSize %f\n", config.gridCellSize);
    printf("movementNoiseScale %f\n", config.movementNoiseScale);
    printf("rotationNoiseScale %f\n", config.rotationNoiseScale);
    printf("velocityDecay %f\n", config.velocityDecay);
    printf("\n");

    if (config.gridSize < config.interactionDistance) {
        printf("WARNING! The interactionDistance (%f) is less than gridCellSize (%f).\nNot all interactions may play out as intended\n", config.interactionDistance, config.gridCellSize);
    }

    /*int sharedMemSize;
    cudaDeviceGetAttribute(&sharedMemSize, cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("sharedMemSize %d\n", sharedMemSize);
    return;*/

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    std::mt19937 rngGen(42);
    std::uniform_real_distribution<double> rng(0.0, 1.0);

    size_t count = config.numParticles;
    size_t size = count * sizeof(Particle);

    printf("[Memory structure]\n");
    printf("Particle size: %d\n", sizeof(Particle));
    printf("Offsets:");
    printf(" pos %d", offsetof(Particle, pos));
    printf(", rot %d", offsetof(Particle, rot));
    printf(", velocity %d", offsetof(Particle, velocity));
    printf(", type %d", offsetof(Particle, type));
    printf(", flags %d", offsetof(Particle, flags));
    printf("\n\n");

    Particle *h_Particles, *h_NextParticles;
    Particle *d_Particles = NULL,
        *d_NextParticles = NULL;
    universalAlloc(&h_Particles, &d_Particles, count);
    universalAlloc(&h_NextParticles, &d_NextParticles, count);
    int *h_nActiveParticles = new int[] { config.numParticles },
        *h_lastActiveParticle = new int[] { config.numParticles-1 };
    int *d_nActiveParticles = NULL,
        *d_lastActiveParticle = NULL;
    unsigned int *d_Indices = NULL,
        *d_NextIndices = NULL,
        *d_GridRanges = NULL;
    cudaAlloc(&d_Indices, count);
    cudaAlloc(&d_NextIndices, count);
    cudaAlloc(&d_GridRanges, config.gridSize * 2);

    cudaAlloc(&d_nActiveParticles, 1, h_nActiveParticles);
    cudaAlloc(&d_lastActiveParticle, 1, h_lastActiveParticle);

    curandState* rngState;
    cudaAlloc(&rngState, count);

    /*Config* d_Config = NULL;
    cudaAlloc(&d_Config, sizeof(Config));
    cudaMemcpy(d_Config, &config, sizeof(Config), cudaMemcpyHostToDevice);*/
    err = cudaMemcpyToSymbol(d_Config, &config, sizeof(Config), 0, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy config to the device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_Indices, d_NextIndices, d_Particles, d_NextParticles, count);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    printf("temp_storage_bytes %d\n", temp_storage_bytes);

    int numTypes = 2;

    // Initialize the host input vectors
    for (int i = 0; i < config.numParticles; i++)
    {
        h_Particles[i].pos = make_float3(
            rng(rngGen) * config.simSize,
            rng(rngGen) * config.simSize,
            rng(rngGen) * config.simSize
        );
        h_Particles[i].rot = make_float4(
            0, 0, 0, 1
        );
        h_Particles[i].velocity = make_float3(0, 0, 0);
        h_Particles[i].type = rng(rngGen) * numTypes;
        h_Particles[i].flags = PARTICLE_FLAG_ACTIVE;
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    err = cudaMemcpy(d_Particles, h_Particles, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int particleBlocksPerGrid = (config.numParticles + threadsPerBlock - 1) / threadsPerBlock;
    printf("Particle CUDA kernel with %d blocks of %d threads\n", particleBlocksPerGrid, threadsPerBlock);

    int gridBlocksPerGrid = (config.gridSize + threadsPerBlock - 1) / threadsPerBlock;
    printf("Grid CUDA kernel with %d blocks of %d threads\n", gridBlocksPerGrid, threadsPerBlock);

    setupKernel KERNEL_ARGS2(particleBlocksPerGrid, threadsPerBlock) (rngState);

    std::ofstream fout;
    fout.open("./results/frames.dat", std::ios::binary | std::ios::out);
    fout.write((char*)&config.simSize, sizeof(float));
    // Particle buffer size
    fout.write((char*)&config.numParticles, sizeof(unsigned int));

    // Number of existing particles
    fout.write((char*)&config.numParticles, sizeof(unsigned int));
    fout.write((char*)h_Particles, size);

    printf("\n");
    printf("[Simulating...]\n");
    
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < config.steps; i++) {
        // Order particles by their grid positions
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        updateIndices KERNEL_ARGS2(particleBlocksPerGrid, threadsPerBlock) (d_Particles, d_Indices);
        cudaDeviceSynchronize();
        std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_Indices, d_NextIndices, d_Particles, d_NextParticles, count);
        cudaDeviceSynchronize();
        std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
        swapBuffers(&d_Particles, &d_NextParticles);
        swapBuffers(&d_Indices, &d_NextIndices);
        //printCUDAIntArray(d_Indices, config.numParticles);
        cudaMemset(d_GridRanges, 0, config.gridSize * 2 * sizeof(unsigned int));
        updateGridRanges KERNEL_ARGS2(gridBlocksPerGrid, threadsPerBlock) (d_Indices, d_GridRanges);
        cudaDeviceSynchronize();

        cudaMemcpy(h_nActiveParticles, d_nActiveParticles, sizeof(int), cudaMemcpyDeviceToHost);
        h_lastActiveParticle[0] = h_nActiveParticles[0];
        cudaMemcpy(d_lastActiveParticle, h_lastActiveParticle, sizeof(int), cudaMemcpyHostToDevice);

        // Simulate the dynamics
        std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
        //printCUDAIntArray(d_GridRanges, config.gridSize * 2);
        move KERNEL_ARGS2(particleBlocksPerGrid, threadsPerBlock) (rngState, d_Particles, d_NextParticles, d_nActiveParticles, d_lastActiveParticle, d_Indices, d_GridRanges);
        swapBuffers(&d_Particles, &d_NextParticles);
        cudaDeviceSynchronize();
        cudaMemcpy(h_nActiveParticles, d_nActiveParticles, sizeof(int), cudaMemcpyDeviceToHost);
        printf("nActiveParticles %d\n", h_nActiveParticles[0]);
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
        fout.write((char*)&config.numParticles, sizeof(unsigned int));
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
    err = cudaMemcpy(h_Particles, d_Particles, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float minX = 0.5;
    for (int i = 0; i < config.numParticles; i++) {
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
    cudaFree(d_nActiveParticles);
    cudaFree(d_lastActiveParticle);
    cudaFree(d_temp_storage);

    // Free host memory
    free(h_Particles);
    free(h_NextParticles);

    printf("Done\n");
    return 0;
}

