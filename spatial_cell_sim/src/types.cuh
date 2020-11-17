#pragma once

#include <cuda_runtime.h>

struct Particle {
    float3 pos;
    float4 rot;
    float3 velocity;
    int type;
    int flags;
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
};
