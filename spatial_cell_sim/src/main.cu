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
#include <random>
#include <ctime>
#include <ratio>
#include <chrono>

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
#include "time.cuh"
#include "macros.cuh"
#include "math.cuh"
#include "grid.cuh"
#include "memory.cuh"
#include "dynamics.cuh"


__global__ void
setupKernel(curandState* rngState) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= d_Config.numParticles)
        return;

    // Setup the random generators
    /*curand_init(42, idx, 0, &rngState[idx]);*/
    // Faster initialization
    // Ref: https://forums.developer.nvidia.com/t/curand-initialization-time/19758/3
    curand_init((42 << 24) + idx, 0, 0, &rngState[idx]);
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

constexpr int threadsPerBlock = 256;

int
numCudaBlocks(int nItems) {
    return (nItems + threadsPerBlock - 1) / threadsPerBlock;
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

    // Load the config values
    Config config;
    config.numParticles = configJson["numParticles"].asInt();
    config.steps = configJson["steps"].asInt();
    config.simSize = configJson["simSize"].asFloat();
    config.interactionDistance = configJson["interactionDistance"].asFloat();
    config.nGridCellsBits = configJson["nGridCellsBits"].asInt();
    config.movementNoiseScale = configJson["movementNoiseScale"].asFloat();
    config.rotationNoiseScale = configJson["rotationNoiseScale"].asFloat();
    config.velocityDecay = configJson["velocityDecay"].asFloat();
    config.relaxationSteps = configJson["relaxationSteps"].asInt();

    // Calculate the derived config values
    config.nGridCells = 1 << config.nGridCellsBits;
    config.gridCellSize = config.simSize / config.nGridCells;
    config.gridSize = config.nGridCells * config.nGridCells * config.nGridCells;

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
    printf("relaxationSteps %d\n", config.relaxationSteps);
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

    // Setup the random generator for the host
    std::mt19937 rngGen(42);
    std::uniform_real_distribution<double> rng(0.0, 1.0);

    size_t count = config.numParticles;
    size_t size = count * sizeof(Particle);

    // Print out the Particle struct's alignment
    printf("[Memory structure]\n");
    printf("Particle size: %d\n", sizeof(Particle));
    printf("Offsets:");
    printf(" id %d", offsetof(Particle, id));
    printf(", type %d", offsetof(Particle, type));
    printf(", flags %d", offsetof(Particle, flags));
    printf(", pos %d", offsetof(Particle, pos));
    printf(", rot %d", offsetof(Particle, rot));
    printf(", velocity %d", offsetof(Particle, velocity));
    printf(", nActiveInteractions %d", offsetof(Particle, nActiveInteractions));
    printf(", interactions %d", offsetof(Particle, interactions));
    printf(", debugVector %d", offsetof(Particle, debugVector));
    printf("\n\n");

    // Allocate the host & device variables
    Particle *h_Particles, *h_NextParticles;
    Particle *d_Particles = NULL,
        *d_NextParticles = NULL;
    universalAlloc(&h_Particles, &d_Particles, count);
    universalAlloc(&h_NextParticles, &d_NextParticles, count);
    int nInactiveParticles = 40;
    int *h_nActiveParticles = new int[] { config.numParticles - nInactiveParticles },
        *h_lastActiveParticle = new int[] { h_nActiveParticles[0]-1 },
        *h_nextParticleId = new int[] { h_nActiveParticles[0] };
    int *d_nActiveParticles = NULL,
        *d_lastActiveParticle = NULL,
        *d_nextParticleId = NULL;
    cudaAlloc(&d_nActiveParticles, 1, h_nActiveParticles);
    cudaAlloc(&d_lastActiveParticle, 1, h_lastActiveParticle);
    cudaAlloc(&d_nextParticleId, 1, h_nextParticleId);
    unsigned int *d_Indices = NULL,
        *d_NextIndices = NULL,
        *d_GridRanges = NULL;
    cudaAlloc(&d_Indices, count);
    cudaAlloc(&d_NextIndices, count);
    cudaAlloc(&d_GridRanges, config.gridSize * 2);

    curandState* rngState;
    cudaAlloc(&rngState, count);

    // Copy the config into the device constant memory
    cudaMemcpyToSymbol(d_Config, &config, sizeof(Config), 0, cudaMemcpyHostToDevice);

    // Allocate the temporary buffer for sorting
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    /*
    * The fix for "uses too much shared data" is to change
    * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include\cub\device\dispatch\dispatch_radix_sort.cuh
    * reduce the number of threads per blocks from 512 to 384, in the `Policy700`, line 788 - the first of "Downsweep policies"
    */
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_Indices, d_NextIndices, d_Particles, d_NextParticles, count);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    printf("temp_storage_bytes %d\n", temp_storage_bytes);

    int numTypes = 2;
    int lastType1ParticleIdx = -1;

    // Initialize the particles
    memset(h_Particles, 0, size);
    for (int i = 0; i < h_nActiveParticles[0]; i++)
    {
        h_Particles[i].id = i;
        h_Particles[i].pos = make_float3(
            rng(rngGen) * config.simSize,
            rng(rngGen) * config.simSize,
            rng(rngGen) * config.simSize
        );
        h_Particles[i].type = rng(rngGen) < 0.05 ? 1 : 0;  //rng(rngGen) * numTypes;
        h_Particles[i].flags = PARTICLE_FLAG_ACTIVE;
        h_Particles[i].rot = random_rotation_host(rng, rngGen);
        h_Particles[i].velocity = VECTOR_ZERO;
        h_Particles[i].nActiveInteractions = 0;

        if (h_Particles[i].type == 1) {
            if (lastType1ParticleIdx >= 0) {
                Particle *interactionPartner = &h_Particles[lastType1ParticleIdx];
                // Position it near the partner particle
                h_Particles[i].pos = make_float3(
                    min(max(interactionPartner->pos.x + 0.0015, 0.0f), 1.0f),
                    min(max(interactionPartner->pos.y + 0.0015, 0.0f), 1.0f),
                    min(max(interactionPartner->pos.z + 0.0015, 0.0f), 1.0f)
                );
                // Add interaction for the partner
                interactionPartner->interactions[interactionPartner->nActiveInteractions].type = 0;
                interactionPartner->interactions[interactionPartner->nActiveInteractions].partnerId = i;
                interactionPartner->nActiveInteractions++;
                // Add interaction for the current particle
                h_Particles[i].interactions[h_Particles[i].nActiveInteractions].type = 0;
                h_Particles[i].interactions[h_Particles[i].nActiveInteractions].partnerId = lastType1ParticleIdx;
                h_Particles[i].nActiveInteractions++;
            }
            else {
                h_Particles[i].pos = make_float3(config.simSize / 4, config.simSize / 4, config.simSize / 4);
            }
            lastType1ParticleIdx = i;
        }

        h_Particles[i].debugVector = make_float4(0, 0, 0, 0);
    }
    copyToDevice(d_Particles, h_Particles, size);

    printf("Particle CUDA kernel with %d blocks of %d threads\n", numCudaBlocks(config.numParticles), threadsPerBlock);

    // Initialize the device-side variables
    setupKernel KERNEL_ARGS2(numCudaBlocks(config.numParticles), threadsPerBlock) (rngState);

    // Write the frames file header
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
    
    // The simulation loop
    time_point t0 = now();
    for (int i = 0; i < config.steps; i++) {
        // Order particles by their grid positions
        time_point t1 = now();
        updateIndices KERNEL_ARGS2(numCudaBlocks(config.numParticles), threadsPerBlock) (d_Particles, d_Indices);
        cudaDeviceSynchronize();
        time_point t3 = now();
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_Indices, d_NextIndices, d_Particles, d_NextParticles, count);
        cudaDeviceSynchronize();
        swapBuffers(&d_Particles, &d_NextParticles);
        swapBuffers(&d_Indices, &d_NextIndices);
        //printCUDAIntArray(d_Indices, config.numParticles);
        time_point t4 = now();
        cudaMemset(d_GridRanges, 0, config.gridSize * 2 * sizeof(unsigned int));
        updateGridRanges KERNEL_ARGS2(numCudaBlocks(config.gridSize), threadsPerBlock) (d_Indices, d_GridRanges);
        cudaDeviceSynchronize();
        //printCUDAIntArray(d_GridRanges, config.gridSize * 2);

        copyToHost(h_nActiveParticles, d_nActiveParticles, sizeof(int));
        h_lastActiveParticle[0] = h_nActiveParticles[0];
        copyToDevice(d_lastActiveParticle, h_lastActiveParticle, sizeof(int));

        // Simulate the dynamics
        time_point t5 = now();
        cudaMemset(d_NextParticles, 0, size);
        move KERNEL_ARGS2(numCudaBlocks(h_nActiveParticles[0]), threadsPerBlock) (i, rngState, d_Particles, d_NextParticles, h_nActiveParticles[0], d_nActiveParticles, d_lastActiveParticle, d_nextParticleId, d_Indices, d_GridRanges);
        swapBuffers(&d_Particles, &d_NextParticles);
        cudaDeviceSynchronize();
        copyToHost(h_nActiveParticles, d_nActiveParticles, sizeof(int));
        time_point t6 = now();

        for (int j = 0; j < config.relaxationSteps; j++) {
            // Relax the accumulated tensions
            cudaMemset(d_NextParticles, 0, size);
            relax KERNEL_ARGS2(numCudaBlocks(h_nActiveParticles[0]), threadsPerBlock) (i, rngState, d_Particles, d_NextParticles, h_nActiveParticles[0], d_Indices, d_GridRanges);
            swapBuffers(&d_Particles, &d_NextParticles);
            cudaDeviceSynchronize();
        }
        time_point t7 = now();

        if (i % 10 == 0) {
            printf("step %d, nActiveParticles %d, updateIndices %f, SortPairs %f, updateGridRanges %f, move %f, relax %f\n",
                i,
                h_nActiveParticles[0],
                getDuration(t1, t3),
                getDuration(t3, t4),
                getDuration(t4, t5),
                getDuration(t5, t6),
                getDuration(t6, t7)
            );
        }

        cudaMemcpy(h_Particles, d_Particles, size, cudaMemcpyDeviceToHost);
        fout.write((char*)&config.numParticles, sizeof(unsigned int));
        fout.write((char*)h_Particles, size);
    }
    cudaDeviceSynchronize();
    fout.close();
    time_point t2 = now();
    printf("time %f\n", getDuration(t0, t2));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaUnalloc(d_Particles);
    cudaUnalloc(d_NextParticles);
    cudaUnalloc(d_Indices);
    cudaUnalloc(d_NextIndices);
    cudaUnalloc(d_GridRanges);
    cudaUnalloc(d_nActiveParticles);
    cudaUnalloc(d_lastActiveParticle);
    cudaUnalloc(d_nextParticleId);
    cudaUnalloc(d_temp_storage);

    // Free host memory
    free(h_Particles);
    free(h_NextParticles);
    delete h_nActiveParticles;
    delete h_lastActiveParticle;
    delete h_nextParticleId;

    printf("Done\n");
    return 0;
}

