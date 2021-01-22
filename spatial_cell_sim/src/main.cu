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
#include <unordered_map>

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
#include "setup/setup.cuh"
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

void
checkForCudaErrors(std::string message) {
    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)! %s\n", cudaGetErrorString(err), message);
        exit(EXIT_FAILURE);
    }
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

    printf("Loading particle type info...\n");
    auto particleTypeInfo = loadParticleTypeInfo();
    printf("Loading complexification info...\n");
    auto complexificationInfo = loadComplexificationInfo();
    // Used for finding interactions by id (it's equivalent to the array index)
    auto flatComplexificationInfo = flattenComplexificationInfo(complexificationInfo);
    flatComplexificationInfo->copyToDevice();
    // Used for finding interactions for a specific molecule type
    auto partnerMappedInteractions = partnerMappedComplexificationInfo(complexificationInfo);
    partnerMappedInteractions.first->copyToDevice();
    partnerMappedInteractions.second->copyToDevice();

    printf("Loading complex info...\n");
    auto complexInfo = loadComplexInfo();

    printf("\n");

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

    // Initialize the particles
    setupParticles(&particles, &nActiveParticles, &lastActiveParticle, &nextParticleId, particleTypeInfo, complexificationInfo, complexInfo, &config, rng);
    // Initialize the metabolic particles
    setupMetabolicParticles(&metabolicParticles, &nActiveMetabolicParticles, particleTypeInfo, &config, rng);

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
    
    // Create initial complex-interactions
    particles.clearNextOnDevice();
    complexify KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
        -1,
        rngState.d_Current,
        particles.d_Current,
        particles.d_Next,
        nActiveParticles.h_Current[0],
        gridRanges.d_Current,
        flatComplexificationInfo->d_Current,
        partnerMappedInteractions.first->d_Current,
        partnerMappedInteractions.second->d_Current
    );
    particles.swap();

    printf("\n");
    printf("[Simulating...]\n");

    double persistFrameDuration = 0.0;
    
    // The simulation loop
    time_point t0 = now();
    for (int step = 0; step < config.steps; step++) {
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

        nActiveParticles.copyToHost();
        lastActiveParticle.h_Current[0] = nActiveParticles.h_Current[0];
        lastActiveParticle.copyToDevice();

        // Simulate the dynamics
        time_point t5 = now();
        particles.clearNextOnDevice();
        move KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            step,
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
            step,
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
            step,
            metabolicParticleRngState.d_Current,
            metabolicParticles.d_Current,
            metabolicParticles.d_Next,
            nActiveMetabolicParticles.h_Current[0],
            metabolicParticleGridRanges.d_Current
        );
        metabolicParticles.swap();

        brownianMovementAndRotation<MetabolicParticle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            step,
            metabolicParticleRngState.d_Current,
            metabolicParticles.d_Current,
            nActiveMetabolicParticles.h_Current[0],
            config.metaboliteMovementNoiseScale
        );

        cudaDeviceSynchronize();
        //nActiveParticles.copyToHost();

        // Co-ordinate noise for interacting partners
        time_point t6_00 = now();
        for (int j = 0; j < config.noiseCoordinationSteps; j++) {
            particles.clearNextOnDevice();

            coordinateNoise KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
                step,
                rngState.d_Current,
                particles.d_Current,
                particles.d_Next,
                nActiveParticles.h_Current[0],
                gridRanges.d_Current,
                (float)j / config.noiseCoordinationSteps
            );
            particles.swap();

            cudaDeviceSynchronize();
        }

        applyNoise<Particle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            step,
            rngState.d_Current,
            particles.d_Current,
            nActiveParticles.h_Current[0]
        );
        cudaDeviceSynchronize();

        time_point t6_1 = now();
        //float stepFraction = 1.0f / config.relaxationSteps;
        for (int j = 0; j < config.relaxationSteps; j++) {
            // Step fraction scales as a decreasing arithmetic progression
            float stepFraction = (config.relaxationSteps - j) * 1.0f / ((1.0f + config.relaxationSteps) * config.relaxationSteps / 2.0f);
            // Relax the accumulated tensions - Particles
            particles.clearNextOnDevice();
            /*applyVelocities<Particle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
                step,
                rngState.d_Current,
                particles.d_Current,
                nActiveParticles.h_Current[0],
                stepFraction
            );*/

            relax KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
                step,
                rngState.d_Current,
                particles.d_Current,
                particles.d_Next,
                nActiveParticles.h_Current[0],
                gridRanges.d_Current,
                flatComplexificationInfo->d_Current
            );
            particles.swap();

            // Relax the accumulated tensions - MetabolicParticles
            metabolicParticles.clearNextOnDevice();
            /*applyVelocities<MetabolicParticle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
                step,
                metabolicParticleRngState.d_Current,
                metabolicParticles.d_Current,
                nActiveMetabolicParticles.h_Current[0],
                stepFraction
            );*/

            if (j % 2 == 0) {
                // Relax the metabolic particles less often
                relaxMetabolicParticles KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
                    step,
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
                    step,
                    rngState.d_Current,
                    particles.d_Current,
                    particles.d_Next,
                    nActiveParticles.h_Current[0],
                    gridRanges.d_Current,
                    metabolicParticles.d_Current,
                    metabolicParticleGridRanges.d_Current
                    );
                particles.swap();
            }

            cudaDeviceSynchronize();
        }

        applyVelocities<Particle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            step,
            rngState.d_Current,
            particles.d_Current,
            nActiveParticles.h_Current[0],
            0.5f
        );
        applyVelocities<MetabolicParticle> KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            step,
            metabolicParticleRngState.d_Current,
            metabolicParticles.d_Current,
            nActiveMetabolicParticles.h_Current[0],
            0.5f
        );
        cudaDeviceSynchronize();

        time_point t6_2 = now();

        // Create complex-interactions
        particles.clearNextOnDevice();
        complexify KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            step,
            rngState.d_Current,
            particles.d_Current,
            particles.d_Next,
            nActiveParticles.h_Current[0],
            gridRanges.d_Current,
            flatComplexificationInfo->d_Current,
            partnerMappedInteractions.first->d_Current,
            partnerMappedInteractions.second->d_Current
        );
        particles.swap();

        // Transition the complex-interactions
        particles.clearNextOnDevice();
        transitionInteractions KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
            step,
            rngState.d_Current,
            particles.d_Current,
            particles.d_Next,
            nActiveParticles.h_Current[0],
            gridRanges.d_Current,
            flatComplexificationInfo->d_Current,
            partnerMappedInteractions.first->d_Current,
            partnerMappedInteractions.second->d_Current
        );
        particles.swap();

        cudaDeviceSynchronize();

        time_point t6_3 = now();
        // Diffuse metabolites a bit faster
        for (int j = 0; j < config.metaboliteDiffusionSteps; j++) {
            metabolicParticles.clearNextOnDevice();
            diffuseMetabolites KERNEL_ARGS2(CUDA_NUM_BLOCKS(nActiveMetabolicParticles.h_Current[0]), CUDA_THREADS_PER_BLOCK) (
                step,
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

        /*particles.copyToHost();
        printf("[");
        for (unsigned int i = 0; i < nActiveParticles.h_Current[0]; i++) {
            printf(" %d", particles.h_Current[i].nActiveInteractions);
        }
        printf(" ]\n");
        break;*/

        time_point t8 = now();

        if (step % config.persistEveryNthFrame == 0) {

            // Wait till the previous frame is persisted
            persistFrameDuration = persistFrameTask.get();

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

            persistFrameTask = std::async(
                persistFrame,
                &reducedParticles,
                &reducedMetabolicParticles,
                &storage
            );
        }

        time_point t9 = now();

        if (step % 10 == 0) {
            printf("step %d", step);
            printf(", nActiveParticles %d", nActiveParticles.h_Current[0]);
            printf(", updateGridAndSort %f", getDuration(t1, t5));
            printf(", move %f", getDuration(t5, t6));
            printf(", moveMetabolicParticles %f", getDuration(t6, t6_00));
            printf(", coordinateNoise %f", getDuration(t6_00, t6_1));
            printf(", relax %f", getDuration(t6_1, t6_2));
            printf(", complexify %f", getDuration(t6_2, t6_3));
            printf(", diffuseMetabolites %f", getDuration(t6_3, t8));
            printf(", persistFrame (previous) %f", persistFrameDuration);
            printf(", reduceParticles %f", getDuration(t8, t9));
            printf(", full step time %f", getDuration(t1, t9));
            printf("\n");
        }
    }
    cudaDeviceSynchronize();

    // Make sure the last frame is persisted
    persistFrameDuration = persistFrameTask.get();

    time_point t2 = now();
    printf("time %f\n", getDuration(t0, t2));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*printf("Press any key to exit...\n");
    getchar();*/

    printf("Done\n");
    return 0;
}

