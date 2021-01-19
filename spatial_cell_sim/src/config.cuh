#pragma once

#include <ctime>
#include <ratio>
#include <chrono>

#include <json/json.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "./constants.cuh"


struct Config {
    int numParticles;
    int numMetabolicParticles;
    int steps;
    float simSize;
    float maxInteractionDistance;
    float maxDiffusionDistance;
    int nGridCellsBits;

    // Derived
    int nGridCells;
    float gridCellSize;
    float gridSize;
    //

    float movementNoiseScale;
    float rotationNoiseScale;
    float metaboliteMovementNoiseScale;
    float velocityDecay;
    float angularVelocityDecay;
    int noiseCoordinationSteps;
    int relaxationSteps;
    int metaboliteDiffusionSteps;

    int persistEveryNthFrame;
    int reportEveryNthFrame;

    Config() {
        //
    }

    Config(Json::Value configJson)
        : numParticles(configJson["numParticles"].asInt()),
        numMetabolicParticles(configJson["numMetabolicParticles"].asInt()),
        steps(configJson["steps"].asInt()),
        simSize(configJson["simSize"].asFloat()),
        maxInteractionDistance(configJson["maxInteractionDistance"].asFloat()),
        maxDiffusionDistance(configJson["maxDiffusionDistance"].asFloat()),
        nGridCellsBits(configJson["nGridCellsBits"].asInt()),
        movementNoiseScale(configJson["movementNoiseScale"].asFloat()),
        rotationNoiseScale(configJson["rotationNoiseScale"].asFloat()),
        metaboliteMovementNoiseScale(configJson["metaboliteMovementNoiseScale"].asFloat()),
        velocityDecay(configJson["velocityDecay"].asFloat()),
        angularVelocityDecay(configJson["angularVelocityDecay"].asFloat()),
        noiseCoordinationSteps(configJson["noiseCoordinationSteps"].asInt()),
        relaxationSteps(configJson["relaxationSteps"].asInt()),
        metaboliteDiffusionSteps(configJson["metaboliteDiffusionSteps"].asInt()),
        persistEveryNthFrame(configJson["persistEveryNthFrame"].asInt()),
        reportEveryNthFrame(configJson["reportEveryNthFrame"].asInt())
    {
        // Calculate the derived values
        nGridCells = 1 << nGridCellsBits;
        gridCellSize = simSize / nGridCells;
        gridSize = nGridCells * nGridCells * nGridCells;
    }

    void
        print() {
        printf("[Config]\n");
        printf("numParticles %d\n", numParticles);
        printf("numMetabolicParticles %d\n", numMetabolicParticles);
        printf("steps %d\n", steps);
        printf("simSize %f\n", simSize);
        printf("maxInteractionDistance %f\n", maxInteractionDistance);
        printf("maxDiffusionDistance %f\n", maxDiffusionDistance);
        printf("nGridCellsBits %d\n", nGridCellsBits);
        printf("nGridCells %d\n", nGridCells);
        printf("gridCellSize %f\n", gridCellSize);
        printf("movementNoiseScale %f\n", movementNoiseScale);
        printf("rotationNoiseScale %f\n", rotationNoiseScale);
        printf("velocityDecay %f\n", velocityDecay);
        printf("angularVelocityDecay %f\n", angularVelocityDecay);
        printf("noiseCoordinationSteps %d\n", noiseCoordinationSteps);
        printf("relaxationSteps %d\n", relaxationSteps);
        printf("metaboliteDiffusionSteps %d\n", metaboliteDiffusionSteps);
        printf("persistEveryNthFrame %d\n", persistEveryNthFrame);
        printf("reportEveryNthFrame %d\n", reportEveryNthFrame);
        printf("\n");
    }
};

__constant__ Config d_Config;
