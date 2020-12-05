#pragma once

#include <random>

#include <cuda_runtime.h>
#include <crt/math_functions.h>

#include "../types.cuh"
#include "../constants.cuh"
#include "../math.cuh"


void
addMetabolitesByCoord(
    int start,
    int end,
    int metaboliteId,
    MetabolicParticle* h_Particles,
    const Config* config,
    std::function<double()> rng
) {
    for (int i = start + 1; i < end; i++) {
        MetabolicParticle* p = &h_Particles[i];
        if (p->pos.x < config->simSize * 0.2) {
            h_Particles[i].metabolites[metaboliteId] = 100.0f;  // *1400.0f / (end - start);
        }
    }
}
