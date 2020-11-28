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
#include <functional>
//#include <windows.h>

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
#include "setup/particle_setup.cuh"
#include "setup/interaction_setup.cuh"


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
    std::uniform_real_distribution<double> rngDist(0.0, 1.0);
    std::function<double()> rng = [&rngDist, &rngGen]() { return rngDist(rngGen); };

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


    //HANDLE hMapFile;
    //LPCTSTR pBuf;
    //std::string memBufName = "spatial_cell_buf";
    //unsigned int memBufSize = 2 * 1024 * 1024 * 1024;

    //hMapFile = CreateFileMapping(
    //    INVALID_HANDLE_VALUE,    // use paging file
    //    NULL,                    // default security
    //    PAGE_READWRITE,          // read/write access
    //    0,                       // maximum object size (high-order DWORD)
    //    memBufSize,                // maximum object size (low-order DWORD)
    //    memBufName.c_str()                 // name of mapping object
    //);

    //if (hMapFile == NULL)
    //{
    //    printf(TEXT("Could not create file mapping object (%d).\n"),
    //        GetLastError());
    //    return 1;
    //}
    //pBuf = (LPTSTR)MapViewOfFile(hMapFile,   // handle to map object
    //    FILE_MAP_ALL_ACCESS, // read/write permission
    //    0,
    //    0,
    //    memBufSize
    //);

    //if (pBuf == NULL)
    //{
    //    printf(TEXT("Could not map view of file (%d).\n"),
    //        GetLastError());

    //    CloseHandle(hMapFile);

    //    return 1;
    //}

    //FillMemory((PVOID)pBuf, 0, memBufSize);

    // Allocate the host & device variables
    DoubleBuffer<Particle> particles(count);
    DeviceOnlyDoubleBuffer<unsigned int> indices(count);
    DeviceOnlySingleBuffer<unsigned int> gridRanges(config.gridSize * 2);
    /*Particle *h_Particles, *h_NextParticles;
    Particle *particles.d_Current = NULL,
        *particles.d_Next = NULL;
    universalAlloc(&h_Particles, &particles.d_Current, count);
    universalAlloc(&h_NextParticles, &particles.d_Next, count);*/
    //int nInactiveParticles = 40;
    SingleBuffer<int> nActiveParticles(1);
    SingleBuffer<int> lastActiveParticle(1);
    SingleBuffer<int> nextParticleId(1);
    /*int *nActiveParticles.h_Current = new int[] { 0 },
        *h_lastActiveParticle = new int[] { -1 },
        *h_nextParticleId = new int[] { 0 };
    int *nActiveParticles.d_Current = NULL,
        *lastActiveParticle.d_Current = NULL,
        *d_nextParticleId = NULL;*/
    /*unsigned int *d_Indices = NULL,
        *d_NextIndices = NULL,
        *d_GridRanges = NULL;
    cudaAlloc(&d_Indices, count);
    cudaAlloc(&d_NextIndices, count);
    cudaAlloc(&d_GridRanges, config.gridSize * 2);*/

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
        indices.d_Current, indices.d_Next, particles.d_Current, particles.d_Next, count);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    printf("temp_storage_bytes %d\n", temp_storage_bytes);

    int numTypes = 2;
    int lastType1ParticleIdx = -1;

    // Initialize the particles
    fillParticlesUniform(
        config.numParticles * 0.8,
        0,
        particles.h_Current, nActiveParticles.h_Current, &config, rng
    );

    int lineStartIdx = nActiveParticles.h_Current[0];
    fillParticlesStraightLine(
        config.numParticles * 0.05,
        1,
        make_float3(config.simSize / 4, config.simSize / 4, config.simSize / 4),
        make_float3(0.0015, 0.0015, 0.0015),
        particles.h_Current, nActiveParticles.h_Current, &config, rng
    );
    printf("rot %f, %f, %f\n", particles.h_Current[lineStartIdx].rot.x, particles.h_Current[lineStartIdx].rot.y, particles.h_Current[lineStartIdx].rot.z);
    printf("rot %f, %f, %f\n", particles.h_Current[lineStartIdx + 1].rot.x, particles.h_Current[lineStartIdx + 1].rot.y, particles.h_Current[lineStartIdx + 1].rot.z);
    int lineEndIdx = nActiveParticles.h_Current[0];
    linkParticlesSerially(
        lineStartIdx,
        lineEndIdx,
        particles.h_Current, &config, rng
    );
    particles.copyToDevice();

    // Set the reference particle numbers/indices
    lastActiveParticle.h_Current[0] = nActiveParticles.h_Current[0] - 1;
    nextParticleId.h_Current[0] = nActiveParticles.h_Current[0];
    nActiveParticles.copyToDevice();
    lastActiveParticle.copyToDevice();
    nextParticleId.copyToDevice();

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
    fout.write((char*)particles.h_Current, size);

    /*char *memBufPtr = (char*)pBuf;
    CopyMemory((PVOID)memBufPtr, (char*)&config.simSize, sizeof(float));
    memBufPtr += sizeof(float);
    CopyMemory((PVOID)memBufPtr, (char*)&config.numParticles, sizeof(unsigned int));
    memBufPtr += sizeof(unsigned int);

    CopyMemory((PVOID)memBufPtr, (char*)&config.numParticles, sizeof(unsigned int));
    memBufPtr += sizeof(unsigned int);
    CopyMemory((PVOID)memBufPtr, (char*)h_Particles, size);
    memBufPtr += size;*/

    printf("\n");
    printf("[Simulating...]\n");
    
    // The simulation loop
    time_point t0 = now();
    for (int i = 0; i < config.steps; i++) {
        // Order particles by their grid positions
        time_point t1 = now();
        updateIndices KERNEL_ARGS2(numCudaBlocks(config.numParticles), threadsPerBlock) (particles.d_Current, indices.d_Current);
        cudaDeviceSynchronize();
        time_point t3 = now();
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            indices.d_Current, indices.d_Next, particles.d_Current, particles.d_Next, count);
        cudaDeviceSynchronize();
        particles.swap();
        indices.swap();
        //printCUDAIntArray(d_Indices, config.numParticles);
        time_point t4 = now();
        gridRanges.clear();
        updateGridRanges KERNEL_ARGS2(numCudaBlocks(config.gridSize), threadsPerBlock) (indices.d_Current, gridRanges.d_Current);
        cudaDeviceSynchronize();
        //printCUDAIntArray(gridRanges.d_Current, config.gridSize * 2);

        nActiveParticles.copyToHost();
        lastActiveParticle.h_Current[0] = nActiveParticles.h_Current[0];
        lastActiveParticle.copyToDevice();

        // Simulate the dynamics
        time_point t5 = now();

        particles.clearNextOnDevice();
        move KERNEL_ARGS2(numCudaBlocks(nActiveParticles.h_Current[0]), threadsPerBlock) (
            i,
            rngState,
            particles.d_Current,
            particles.d_Next,
            nActiveParticles.h_Current[0],
            nActiveParticles.d_Current,
            lastActiveParticle.d_Current,
            nextParticleId.d_Current,
            indices.d_Current,
            gridRanges.d_Current
        );
        particles.swap();
        cudaDeviceSynchronize();
        copyToHost(nActiveParticles.h_Current, nActiveParticles.d_Current, sizeof(int));
        time_point t6 = now();

        for (int j = 0; j < config.relaxationSteps; j++) {
            // Relax the accumulated tensions
            cudaMemset(particles.d_Next, 0, size);
            relax KERNEL_ARGS2(numCudaBlocks(nActiveParticles.h_Current[0]), threadsPerBlock) (
                i,
                rngState,
                particles.d_Current,
                particles.d_Next,
                nActiveParticles.h_Current[0],
                indices.d_Current,
                gridRanges.d_Current
            );
            particles.swap();
            cudaDeviceSynchronize();
        }
        time_point t7 = now();

        particles.copyToHost();

        time_point t8 = now();

        fout.write((char*)&config.numParticles, sizeof(unsigned int));
        fout.write((char*)particles.h_Current, size);

        /*CopyMemory((PVOID)memBufPtr, (char*)&config.numParticles, sizeof(unsigned int));
        memBufPtr += sizeof(unsigned int);
        CopyMemory((PVOID)memBufPtr, (char*)h_Particles, size);
        memBufPtr += size;*/

        time_point t9 = now();

        if (i % 10 == 0) {
            printf("step %d, nActiveParticles %d, updateIndices %f, SortPairs %f, updateGridRanges %f, move %f, relax %f, cudaMemcpy %f, fout.write %f\n",
                i,
                nActiveParticles.h_Current[0],
                getDuration(t1, t3),
                getDuration(t3, t4),
                getDuration(t4, t5),
                getDuration(t5, t6),
                getDuration(t6, t7),
                getDuration(t7, t8),
                getDuration(t8, t9)
            );
        }
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

    /*printf("Press any key to exit...\n");
    getchar();

    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);*/

    /*cudaUnalloc(particles.d_Current);
    cudaUnalloc(particles.d_Next);*/
    /*cudaUnalloc(d_Indices);
    cudaUnalloc(d_NextIndices);*/
    /*cudaUnalloc(d_GridRanges);
    cudaUnalloc(nActiveParticles.d_Current);
    cudaUnalloc(lastActiveParticle.d_Current);
    cudaUnalloc(d_nextParticleId);*/
    cudaUnalloc(d_temp_storage);

    // Free host memory
    /*free(h_Particles);
    free(h_NextParticles);*/
    /*delete nActiveParticles.h_Current;
    delete h_lastActiveParticle;
    delete h_nextParticleId;*/

    printf("Done\n");
    return 0;
}

