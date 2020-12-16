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
#include <future>
#include <vector>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <crt/math_functions.h>

#include <json/json.h>

#include "types.cuh"
#include "constants.cuh"
#include "config.cuh"
#include "time.cuh"
#include "macros.cuh"
#include "math.cuh"
#include "grid.cuh"
#include "memory.cuh"
#include "storage.cuh"
#include "chemistry.cuh"
#include "dynamics.cuh"
#include "metabolites.cuh"
#include "setup/particle_setup.cuh"
#include "setup/interaction_setup.cuh"
#include "setup/metabolite_setup.cuh"


__global__ void
setupRandomDevice(curandState* rngState, int numItems) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= numItems)
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

double
persistFrame(
    SingleBuffer<ReducedParticle> *particles,
    SingleBuffer<ReducedMetabolicParticle> *metabolicParticles,
    FileStorage *storage
) {
    time_point t1 = now();
    particles->copyToHost();
    metabolicParticles->copyToHost();
    storage->writeFrame<ReducedParticle, ReducedMetabolicParticle>(particles, metabolicParticles);
    time_point t2 = now();
    return getDuration(t1, t2);
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
    Config config(configJson);
    config.print();

    if (config.gridSize < config.maxInteractionDistance) {
        printf("WARNING! The maxInteractionDistance (%f) is less than gridCellSize (%f).\nNot all interactions may play out as intended\n", config.maxInteractionDistance, config.gridCellSize);
    }
    if (config.gridSize < config.maxDiffusionDistance) {
        printf("WARNING! The maxDiffusionDistance (%f) is less than gridCellSize (%f).\nMetabolite diffusion may play out as intended\n", config.maxDiffusionDistance, config.gridCellSize);
    }

    auto particleTypeInfo = loadParticleTypeInfo();

    // TODO: check that the maxDiffusionDistance doesn't span the lipid layer width + metabolite-lipid collision distance

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

    // Print out the Particle struct's alignment
    printf("[Memory structure]\n");
    printf("----- Particle size: %d\n", sizeof(Particle));
    printf("Offsets:");
    printf(" id %d", offsetof(Particle, id));
    printf(", type %d", offsetof(Particle, type));
    printf(", flags %d", offsetof(Particle, flags));
    printf(", radius %d", offsetof(Particle, radius));
    printf(", pos %d", offsetof(Particle, pos));
    printf(", rot %d", offsetof(Particle, rot));
    printf(", velocity %d", offsetof(Particle, velocity));
    printf(", angularVelocity %d", offsetof(Particle, angularVelocity));
    printf(", nActiveInteractions %d", offsetof(Particle, nActiveInteractions));
    printf(", interactions %d", offsetof(Particle, interactions));
    printf(", debugVector %d", offsetof(Particle, debugVector));
    printf("\n");
    printf("----- MetabolicParticle size: %d\n", sizeof(MetabolicParticle));
    printf("Offsets:");
    printf(" metabolites %d", offsetof(MetabolicParticle, metabolites));
    printf("\n");
    printf("----- ReducedParticle size: %d\n", sizeof(ReducedParticle));
    printf("Offsets:");
    printf(" id %d", offsetof(ReducedParticle, id));
    printf(", type %d", offsetof(ReducedParticle, type));
    printf(", flags %d", offsetof(ReducedParticle, flags));
    printf(", radius %d", offsetof(ReducedParticle, radius));
    /*printf(", pos %d", offsetof(ReducedParticle, pos));
    printf(", rot %d", offsetof(ReducedParticle, rot));*/
    printf(", posX %d", offsetof(ReducedParticle, posX));
    printf(", posY %d", offsetof(ReducedParticle, posY));
    printf(", posZ %d", offsetof(ReducedParticle, posZ));
    printf(", rotX %d", offsetof(ReducedParticle, rotX));
    printf(", rotY %d", offsetof(ReducedParticle, rotY));
    printf(", rotZ %d", offsetof(ReducedParticle, rotZ));
    printf(", rotW %d", offsetof(ReducedParticle, rotW));
    //printf(", debugVector %d", offsetof(ReducedParticle, debugVector));
    printf("\n");
    printf("----- ReducedMetabolicParticle size: %d\n", sizeof(ReducedMetabolicParticle));
    printf("Offsets:");
    printf(" metabolites %d", offsetof(ReducedMetabolicParticle, metabolites));
    printf("\n\n");

    std::string storageFileName = "./results/frames.dat";
    FileStorage storage(storageFileName, &config);


    // TODO: implement using boost's shared memory: https://www.boost.org/doc/libs/1_74_0/doc/html/interprocess/quick_guide.html
    
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
    DoubleBuffer<Particle> particles(config.numParticles);
    SingleBuffer<ReducedParticle> reducedParticles(config.numParticles);
    DeviceOnlyDoubleBuffer<unsigned int> indices(config.numParticles);
    DeviceOnlySingleBuffer<unsigned int> gridRanges(config.gridSize * 2);
    SingleBuffer<int> nActiveParticles(1);
    SingleBuffer<int> lastActiveParticle(1);
    SingleBuffer<int> nextParticleId(1);
    RadixSortPairs<Particle> particleSort(&indices, &particles);

    DoubleBuffer<MetabolicParticle> metabolicParticles(config.numMetabolicParticles);
    SingleBuffer<ReducedMetabolicParticle> reducedMetabolicParticles(config.numMetabolicParticles);
    DeviceOnlyDoubleBuffer<unsigned int> metabolicParticleIndices(config.numMetabolicParticles);
    DeviceOnlySingleBuffer<unsigned int> metabolicParticleGridRanges(config.gridSize * 2);
    SingleBuffer<int> nActiveMetabolicParticles(1);
    RadixSortPairs<MetabolicParticle> metabolicParticleSort(&metabolicParticleIndices, &metabolicParticles);

    DeviceOnlySingleBuffer<curandState> rngState(config.numParticles);
    DeviceOnlySingleBuffer<curandState> metabolicParticleRngState(config.numMetabolicParticles);

    // Copy the config into the device constant memory
    cudaMemcpyToSymbol(d_Config, &config, sizeof(Config), 0, cudaMemcpyHostToDevice);

    int numTypes = 2;
    int lastType1ParticleIdx = -1;

    // Initialize the particles
    fillParticlesUniform<Particle>(
        config.numParticles * 0.1,
        PARTICLE_TYPE_DNA,
        particles.h_Current, nActiveParticles.h_Current, particleTypeInfo, &config, rng
    );
    /*int lineStartIdx = nActiveParticles.h_Current[0];
    fillParticlesStraightLine<Particle>(
        config.numParticles * 0.05,
        PARTICLE_TYPE_DNA,
        make_float3(config.simSize / 4, config.simSize / 4, config.simSize / 4),
        make_float3(0.0015, 0.0015, 0.0015),
        particles.h_Current, nActiveParticles.h_Current, particleTypeInfo, &config, rng
    );
    int lineEndIdx = nActiveParticles.h_Current[0];
    linkParticlesSerially<Particle>(
        lineStartIdx,
        lineEndIdx,
        0,
        particles.h_Current, &config, rng
    );*/
    /*fillParticlesPlane<Particle>(
        sqrt(config.numParticles * 0.4),
        PARTICLE_TYPE_LIPID,
        make_float3(0.35 * config.simSize, 0.5 * config.simSize, 0.5 * config.simSize),
        make_float3(-1, 0, 0),
        particles.h_Current, nActiveParticles.h_Current, particleTypeInfo, &config, rng
    );*/

    /*fillParticlesSphere(
        config.numParticles * 0.35,
        PARTICLE_TYPE_LIPID,
        make_float3(0.5 * config.simSize, 0.5 * config.simSize, 0.5 * config.simSize),
        particles.h_Current, nActiveParticles.h_Current, particleTypeInfo, &config, rng
    );*/
    std::vector<int> chainMembers;
    for (int i = 0; i < 3; i++) {
        for (auto it = particleTypeInfo->begin(); it != particleTypeInfo->end(); it++) {
            if (it->second.category == "rna")
                chainMembers.insert(chainMembers.end(), it->first);
            /*if (chainMembers.size() >= 3)
                break;*/
        }
    }
    int chainStartIdx = nActiveParticles.h_Current[0];
    fillParticlesWrappedChain(
        &chainMembers,
        make_float3(0.5 * config.simSize, 0.5 * config.simSize, 0.5 * config.simSize),
        particles.h_Current, nActiveParticles.h_Current, particleTypeInfo, &config, rng
    );
    int chainEndIdx = nActiveParticles.h_Current[0];
    linkParticlesSerially<Particle>(
        chainStartIdx,
        chainEndIdx,
        1,
        particles.h_Current, &config, rng
    );

    /*fillParticlesSphere(
        config.numParticles * 0.23,
        PARTICLE_TYPE_DNA,
        make_float3(0.5 * config.simSize, 0.5 * config.simSize, 0.5 * config.simSize),
        particles.h_Current, nActiveParticles.h_Current, particleTypeInfo, &config, rng
    );*/

    /*fillParticlesSphere(
        config.numParticles * 0.1,
        PARTICLE_TYPE_LIPID,
        make_float3(0.2 * config.simSize, 0.5 * config.simSize, 0.5 * config.simSize),
        particles.h_Current, nActiveParticles.h_Current, particleTypeInfo, &config, rng
    );
    fillParticlesSphere(
        config.numParticles * 0.01,
        PARTICLE_TYPE_DNA,
        make_float3(0.2 * config.simSize, 0.5 * config.simSize, 0.5 * config.simSize),
        particles.h_Current, nActiveParticles.h_Current, particleTypeInfo, &config, rng
    );*/
    /*fillParticlesSphere(
        config.numParticles * 0.15,
        PARTICLE_TYPE_DNA,
        make_float3(0.5 * config.simSize, 0.5 * config.simSize, 0.5 * config.simSize),
        particles.h_Current, nActiveParticles.h_Current, particleTypeInfo, &config, rng
    );
    fillParticlesSphere(
        config.numParticles * 0.08,
        PARTICLE_TYPE_DNA,
        make_float3(0.5 * config.simSize, 0.5 * config.simSize, 0.5 * config.simSize),
        particles.h_Current, nActiveParticles.h_Current, particleTypeInfo, &config, rng
    );*/
    /*fillParticlesSphere(
        config.numParticles * 0.05,
        PARTICLE_TYPE_DNA,
        make_float3(0.5 * config.simSize, 0.5 * config.simSize, 0.5 * config.simSize),
        particles.h_Current, nActiveParticles.h_Current, particleTypeInfo, &config, rng
    );*/
    particles.copyToDevice();

    // Initialize the metabolic particles
    fillParticlesUniform<MetabolicParticle>(
        config.numMetabolicParticles,
        PARTICLE_TYPE_METABOLIC,
        metabolicParticles.h_Current, nActiveMetabolicParticles.h_Current, particleTypeInfo, &config, rng
    );
    addMetabolitesByCoord(
        0,
        config.numMetabolicParticles,
        0,
        metabolicParticles.h_Current, &config, rng
    );
    metabolicParticles.copyToDevice();

    // Set the reference particle numbers/indices
    lastActiveParticle.h_Current[0] = nActiveParticles.h_Current[0] - 1;
    nextParticleId.h_Current[0] = nActiveParticles.h_Current[0];
    nActiveParticles.copyToDevice();
    lastActiveParticle.copyToDevice();
    nextParticleId.copyToDevice();

    nActiveMetabolicParticles.copyToDevice();

    printf("Particle CUDA kernels with %d blocks of %d threads\n", CUDA_NUM_BLOCKS(config.numParticles), CUDA_THREADS_PER_BLOCK);
    printf("MetabolicParticle CUDA kernels with %d blocks of %d threads\n", CUDA_NUM_BLOCKS(config.numMetabolicParticles), CUDA_THREADS_PER_BLOCK);

    // Initialize the device-side variables
    setupRandomDevice KERNEL_ARGS2(CUDA_NUM_BLOCKS(config.numParticles), CUDA_THREADS_PER_BLOCK) (
        rngState.d_Current,
        config.numParticles
    );
    setupRandomDevice KERNEL_ARGS2(CUDA_NUM_BLOCKS(config.numMetabolicParticles), CUDA_THREADS_PER_BLOCK) (
        metabolicParticleRngState.d_Current,
        config.numMetabolicParticles
    );

    // Write the frames file header
    storage.writeHeader();

    // Initial grid-sorting of particles
    updateGridAndSort(
        &particles,
        &indices,
        &gridRanges,
        &particleSort,
        config.numParticles,
        &config
    );
    updateGridAndSort(
        &metabolicParticles,
        &metabolicParticleIndices,
        &metabolicParticleGridRanges,
        &metabolicParticleSort,
        config.numMetabolicParticles,
        &config
    );
    cudaDeviceSynchronize();
    // Remove the initially interfering metabolic particles
    metabolicParticles.clearNextOnDevice();
    removeInterferingMetabolicParticles KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
        metabolicParticles.d_Current,
        metabolicParticles.d_Next,
        nActiveMetabolicParticles.h_Current[0],
        metabolicParticleGridRanges.d_Current,
        particles.d_Current,
        gridRanges.d_Current
    );
    metabolicParticles.swap();
    cudaDeviceSynchronize();
    metabolicParticles.copyToHost();

    // Reduce particles buffer to slimmer representation for saving
    reduceParticles<Particle, ReducedParticle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
        particles.d_Current,
        reducedParticles.d_Current,
        nActiveParticles.h_Current[0]
    );
    reduceParticles<MetabolicParticle, ReducedMetabolicParticle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
        metabolicParticles.d_Current,
        reducedMetabolicParticles.d_Current,
        nActiveMetabolicParticles.h_Current[0]
    );
    cudaDeviceSynchronize();

    // Write the first frame
    std::future<double> persistFrameTask = std::async(
        persistFrame,
        &reducedParticles,
        &reducedMetabolicParticles,
        &storage
    );
    /*reducedParticles.copyToHost();
    reducedMetabolicParticles.copyToHost();
    storage.writeFrame<ReducedParticle, ReducedMetabolicParticle>(&reducedParticles, &reducedMetabolicParticles);*/

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
        updateGridAndSort(
            &particles,
            &indices,
            &gridRanges,
            &particleSort,
            config.numParticles,
            &config
        );
        // Same for the metabolic particles
        updateGridAndSort(
            &metabolicParticles,
            &metabolicParticleIndices,
            &metabolicParticleGridRanges,
            &metabolicParticleSort,
            config.numMetabolicParticles,
            &config
        );
        cudaDeviceSynchronize();

        //updateIndices KERNEL_ARGS2(CUDA_NUM_BLOCKS(config.numParticles), CUDA_THREADS_PER_BLOCK) (particles.d_Current, indices.d_Current);
        //cudaDeviceSynchronize();
        //time_point t3 = now();
        //particleSort.sort(&indices, &particles);
        //cudaDeviceSynchronize();
        //particles.swap();
        //indices.swap();
        ////printCUDAIntArray(d_Indices, config.numParticles);
        //time_point t4 = now();
        //gridRanges.clear();
        //updateGridRanges KERNEL_ARGS2(CUDA_NUM_BLOCKS(config.gridSize), CUDA_THREADS_PER_BLOCK) (indices.d_Current, gridRanges.d_Current);
        //cudaDeviceSynchronize();
        ////printCUDAIntArray(gridRanges.d_Current, config.gridSize * 2);

        nActiveParticles.copyToHost();
        lastActiveParticle.h_Current[0] = nActiveParticles.h_Current[0];
        lastActiveParticle.copyToDevice();

        // Simulate the dynamics
        time_point t5 = now();
        particles.clearNextOnDevice();
        move KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            i,
            rngState.d_Current,
            particles.d_Current,
            particles.d_Next,
            nActiveParticles.h_Current[0],
            nActiveParticles.d_Current,
            lastActiveParticle.d_Current,
            nextParticleId.d_Current,
            gridRanges.d_Current
        );
        particles.swap();

        brownianMovementAndRotation<Particle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            i,
            rngState.d_Current,
            particles.d_Current,
            nActiveParticles.h_Current[0],
            config.movementNoiseScale
        );

        cudaDeviceSynchronize();

        nActiveParticles.copyToHost();

        time_point t6 = now();
        metabolicParticles.clearNextOnDevice();
        moveMetabolicParticles KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            i,
            metabolicParticleRngState.d_Current,
            metabolicParticles.d_Current,
            metabolicParticles.d_Next,
            nActiveMetabolicParticles.h_Current[0],
            metabolicParticleGridRanges.d_Current
        );
        metabolicParticles.swap();

        brownianMovementAndRotation<MetabolicParticle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            i,
            metabolicParticleRngState.d_Current,
            metabolicParticles.d_Current,
            nActiveMetabolicParticles.h_Current[0],
            config.metaboliteMovementNoiseScale
        );

        cudaDeviceSynchronize();
        //nActiveParticles.copyToHost();

        time_point t6_1 = now();
        float stepFraction = 1.0f / config.relaxationSteps;
        for (int j = 0; j < config.relaxationSteps; j++) {
            // Relax the accumulated tensions - Particles
            particles.clearNextOnDevice();
            applyVelocities<Particle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
                i,
                rngState.d_Current,
                particles.d_Current,
                nActiveParticles.h_Current[0],
                stepFraction
            );

            relax KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
                i,
                rngState.d_Current,
                particles.d_Current,
                particles.d_Next,
                nActiveParticles.h_Current[0],
                gridRanges.d_Current
            );
            particles.swap();

            // Relax the accumulated tensions - MetabolicParticles
            metabolicParticles.clearNextOnDevice();
            applyVelocities<MetabolicParticle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
                i,
                metabolicParticleRngState.d_Current,
                metabolicParticles.d_Current,
                nActiveMetabolicParticles.h_Current[0],
                stepFraction
            );

            relaxMetabolicParticles KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
                i,
                metabolicParticleRngState.d_Current,
                metabolicParticles.d_Current,
                metabolicParticles.d_Next,
                nActiveMetabolicParticles.h_Current[0],
                metabolicParticleGridRanges.d_Current,
                particles.d_Current,
                gridRanges.d_Current
            );
            metabolicParticles.swap();

            // Relax the metabolic-plain particle tensions
            particles.clearNextOnDevice();
            relaxMetabolicParticlePartners KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
                i,
                rngState.d_Current,
                particles.d_Current,
                particles.d_Next,
                nActiveParticles.h_Current[0],
                gridRanges.d_Current,
                metabolicParticles.d_Current,
                metabolicParticleGridRanges.d_Current
            );
            particles.swap();

            cudaDeviceSynchronize();
        }

        time_point t6_2 = now();
        //for (int j = 0; j < config.relaxationSteps; j++) {
        //    // Relax the accumulated metabolic particle tensions
        //    
        //    cudaDeviceSynchronize();
        //}

        time_point t6_3 = now();
        // Diffuse metabolites a bit faster
        for (int j = 0; j < config.metaboliteDiffusionSteps; j++) {
            metabolicParticles.clearNextOnDevice();
            diffuseMetabolites KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
                i,
                metabolicParticleRngState.d_Current,
                metabolicParticles.d_Current,
                metabolicParticles.d_Next,
                nActiveMetabolicParticles.h_Current[0],
                metabolicParticleGridRanges.d_Current
            );
            metabolicParticles.swap();
            cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();

        time_point t7 = now();
        /*particles.copyToHost();
        metabolicParticles.copyToHost();*/

        time_point t8 = now();

        // Wait till the previous frame is persisted
        double persistFrameDuration = persistFrameTask.get();

        reduceParticles<Particle, ReducedParticle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            particles.d_Current,
            reducedParticles.d_Current,
            nActiveParticles.h_Current[0]
        );
        //reducedParticles.copyToHost();
        reduceParticles<MetabolicParticle, ReducedMetabolicParticle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            metabolicParticles.d_Current,
            reducedMetabolicParticles.d_Current,
            nActiveMetabolicParticles.h_Current[0]
        );
        //reducedMetabolicParticles.copyToHost();
        //storage.writeFrame<ReducedParticle, ReducedMetabolicParticle>(&reducedParticles, &reducedMetabolicParticles);
        persistFrameTask = std::async(
            persistFrame,
            &reducedParticles,
            &reducedMetabolicParticles,
            &storage
        );

        /*CopyMemory((PVOID)memBufPtr, (char*)&config.numParticles, sizeof(unsigned int));
        memBufPtr += sizeof(unsigned int);
        CopyMemory((PVOID)memBufPtr, (char*)h_Particles, size);
        memBufPtr += size;*/

        time_point t9 = now();

        if (i % 10 == 0) {
            //time_point t7 = now();
            //particles.copyToHost();
            //metabolicParticles.copyToHost();

            //time_point t8 = now();

            //reduceParticles<Particle, ReducedParticle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            //    particles.d_Current,
            //    reducedParticles.d_Current,
            //    nActiveParticles.h_Current[0]
            //    );
            //reducedParticles.copyToHost();
            //reduceParticles<MetabolicParticle, ReducedMetabolicParticle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            //    metabolicParticles.d_Current,
            //    reducedMetabolicParticles.d_Current,
            //    nActiveMetabolicParticles.h_Current[0]
            //    );
            //reducedMetabolicParticles.copyToHost();

            //storage.writeFrame<ReducedParticle, ReducedMetabolicParticle>(&reducedParticles, &reducedMetabolicParticles);

            ///*CopyMemory((PVOID)memBufPtr, (char*)&config.numParticles, sizeof(unsigned int));
            //memBufPtr += sizeof(unsigned int);
            //CopyMemory((PVOID)memBufPtr, (char*)h_Particles, size);
            //memBufPtr += size;*/

            //time_point t9 = now();

            printf("step %d, nActiveParticles %d, updateGridAndSort %f, move %f, moveMetabolicParticles %f, relax %f, diffuseMetabolites %f, persistFrame (previous) %f, reduceParticles %f, full step time %f\n",
                i,
                nActiveParticles.h_Current[0],
                getDuration(t1, t5),
                getDuration(t5, t6),
                getDuration(t6, t6_1),
                getDuration(t6_1, t6_2),
                getDuration(t6_3, t7),
                persistFrameDuration,
                getDuration(t8, t9),
                getDuration(t1, t9)
            );
        }
    }
    cudaDeviceSynchronize();

    // Make sure the last frame is persisted
    double persistFrameDuration = persistFrameTask.get();

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

    printf("Done\n");
    return 0;
}

