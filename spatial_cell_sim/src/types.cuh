#pragma once

#include <ctime>
#include <ratio>
#include <chrono>

#include <json/json.h>

#include <cuda_runtime.h>

#include "./constants.cuh"


typedef std::chrono::high_resolution_clock::time_point time_point;

struct ParticleInteraction {
    int type;
    int partnerId;
};

struct Particle {
    int id;
    int type;
    int flags;
    float3 pos;
    float4 rot;
    float3 velocity;
    int nActiveInteractions;
    ParticleInteraction interactions[4];
    float4 debugVector;

    __device__ __host__ Particle(
        int id = 0,
        int type = 0,
        int flags = 0,
        float3 pos = make_float3(0, 0, 0),
        float4 rot = make_float4(0, 0, 0, 1),
        float3 velocity = make_float3(0, 0, 0)
    ) : id(id),
        type(type),
        flags(flags | PARTICLE_FLAG_ACTIVE),
        pos(pos),
        rot(rot),
        velocity(velocity),
        nActiveInteractions(0),
        interactions(),
        debugVector(make_float4(0, 0, 0, 0))
    {
        //
    }
};

struct MetabolicParticle : Particle {
    float metabolites[NUM_METABOLITES];

    //using Particle::Particle;
    __device__ __host__ MetabolicParticle(
        int id = 0,
        int type = 0,
        int flags = 0,
        float3 pos = make_float3(0, 0, 0),
        float4 rot = make_float4(0, 0, 0, 1),
        float3 velocity = make_float3(0, 0, 0)
    ) : Particle(id, type, flags, pos, rot, velocity)
    {
        memset(metabolites, 0, NUM_METABOLITES * sizeof(float));
    }
};

struct Config {
    int numParticles;
    int numMetabolicParticles;
    int steps;
    float simSize;
    float interactionDistance;
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
    int relaxationSteps;

    Config() {
        //
    }

    Config(Json::Value configJson)
        : numParticles(configJson["numParticles"].asInt()),
          numMetabolicParticles(configJson["numMetabolicParticles"].asInt()),
          steps(configJson["steps"].asInt()),
          simSize(configJson["simSize"].asFloat()),
          interactionDistance(configJson["interactionDistance"].asFloat()),
          nGridCellsBits(configJson["nGridCellsBits"].asInt()),
          movementNoiseScale(configJson["movementNoiseScale"].asFloat()),
          rotationNoiseScale(configJson["rotationNoiseScale"].asFloat()),
          metaboliteMovementNoiseScale(configJson["metaboliteMovementNoiseScale"].asFloat()),
          velocityDecay(configJson["velocityDecay"].asFloat()),
          relaxationSteps(configJson["relaxationSteps"].asInt())
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
        printf("interactionDistance %f\n", interactionDistance);
        printf("nGridCellsBits %d\n", nGridCellsBits);
        printf("nGridCells %d\n", nGridCells);
        printf("gridCellSize %f\n", gridCellSize);
        printf("movementNoiseScale %f\n", movementNoiseScale);
        printf("rotationNoiseScale %f\n", rotationNoiseScale);
        printf("velocityDecay %f\n", velocityDecay);
        printf("relaxationSteps %d\n", relaxationSteps);
        printf("\n");
    }
};

__constant__ Config d_Config;
