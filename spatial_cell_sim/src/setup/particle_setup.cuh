#pragma once

#include <random>

#include <cuda_runtime.h>
#include <crt/math_functions.h>

#include "../types.cuh"
#include "../constants.cuh"
#include "../math.cuh"


void
fillParticlesUniform(
	int count,
	int type,
	Particle* h_Particles,
	int *h_nActiveParticles,
    const Config *config,
    std::function<double()> rng
) {
    printf("fillParticlesUniform %d\n", count);
    for (int i = h_nActiveParticles[0]; i < h_nActiveParticles[0] + count; i++)
    {
        h_Particles[i] = Particle(
            i,
            type,
            0,
            make_float3(
                rng() * config->simSize,
                rng() * config->simSize,
                rng() * config->simSize
            ),
            random_rotation_host(rng)
        );
    }
    h_nActiveParticles[0] += count;
}

void
fillParticlesStraightLine(
    int count,
    int type,
    float3 startPos,
    float3 dPos,
    Particle* h_Particles,
    int* h_nActiveParticles,
    const Config* config,
    std::function<double()> rng
) {
    for (int i = h_nActiveParticles[0]; i < h_nActiveParticles[0] + count; i++)
    {
        float3 pos = add(startPos, mul(dPos, i - h_nActiveParticles[0]));
        h_Particles[i] = Particle(
            i,
            type,
            0,
            make_float3(
                min(max(pos.x, 0.0f), 1.0f),
                min(max(pos.y, 0.0f), 1.0f),
                min(max(pos.z, 0.0f), 1.0f)
            ),
            random_rotation_host(rng)
        );
    }
    h_nActiveParticles[0] += count;
}
