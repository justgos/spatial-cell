#pragma once

#include <ctime>
#include <ratio>
#include <chrono>

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
    float metabolites[1000];

    using Particle::Particle;
};

struct Config {
    int numParticles;
    int steps;
    float simSize;
    float interactionDistance;
    int nGridCellsBits;

    // Computed
    int nGridCells;
    float gridCellSize;
    float gridSize;

    float movementNoiseScale;
    float rotationNoiseScale;
    float velocityDecay;
    int relaxationSteps;
};

__constant__ Config d_Config;
