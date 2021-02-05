#pragma once

#include <random>

#include <cuda_runtime.h>
#include <crt/math_functions.h>

#include "../types.cuh"
#include "../constants.cuh"
#include "../math.cuh"

#include "./particle_setup.cuh"
#include "./interaction_setup.cuh"
#include "./metabolite_setup.cuh"


void
setupParticles(
    DoubleBuffer<Particle> *particles,
    SingleBuffer<int> *nActiveParticles,
    SingleBuffer<int> *lastActiveParticle,
    SingleBuffer<int> *nextParticleId,
    std::unordered_map<int, ParticleTypeInfo>* particleTypeInfo,
    std::unordered_map<int, ParticleInteractionInfo>* complexificationInfo,
    std::unordered_map<int, ComplexInfo> *complexInfo,
    const Config* config,
    std::function<double()> rng
) {
    fillParticlesUniform<Particle>(
        config->numParticles * 0.1,
        PARTICLE_TYPE_DNA,
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );
    /*int lineStartIdx = nActiveParticles->h_Current[0];
    fillParticlesStraightLine<Particle>(
        config->numParticles * 0.05,
        PARTICLE_TYPE_DNA,
        make_float3(config->simSize / 4, config->simSize / 4, config->simSize / 4),
        make_float3(0.0015, 0.0015, 0.0015),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );
    int lineEndIdx = nActiveParticles->h_Current[0];
    linkParticlesSerially<Particle>(
        lineStartIdx,
        lineEndIdx,
        0,
        particles->h_Current, config, rng
    );*/
    /*fillParticlesPlane<Particle>(
        sqrt(config->numParticles * 0.4),
        PARTICLE_TYPE_LIPID,
        make_float3(0.35 * config->simSize, 0.5 * config->simSize, 0.5 * config->simSize),
        make_float3(-1, 0, 0),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );*/

    // Outer lipid half-layer
    fillParticlesSphere(
        pow(44.0f, 2.0f),
        PARTICLE_TYPE_LIPID,
        make_float3(0.5 * config->simSize, 0.5 * config->simSize, 0.7 * config->simSize),
        QUATERNION_IDENTITY,
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );
    // Inner lipid half-layer
    fillParticlesSphere(
        pow(40.0f, 2.0f),
        PARTICLE_TYPE_LIPID,
        make_float3(0.5 * config->simSize, 0.5 * config->simSize, 0.7 * config->simSize),
        quaternion(VECTOR_RIGHT, PI),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );

    std::vector<int> chainMembers;
    for (int i = 0; i < 2; i++) {
        for (auto it = particleTypeInfo->begin(); it != particleTypeInfo->end(); it++) {
            if (it->second.category == "rna")
                chainMembers.insert(chainMembers.end(), it->first);
            /*if (chainMembers.size() >= 3)
                break;*/
        }
    }
    int chainStartIdx = nActiveParticles->h_Current[0];
    fillParticlesWrappedChain(
        &chainMembers,
        make_float3(0.2 * config->simSize, 0.5 * config->simSize, 0.5 * config->simSize),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );
    int chainEndIdx = nActiveParticles->h_Current[0];
    linkParticlesSerially<Particle>(
        chainStartIdx,
        chainEndIdx,
        1,
        particles->h_Current, complexificationInfo, config, rng
    );

    instantiateComplex(
        1,
        make_float3(0.5 * config->simSize, 0.5 * config->simSize, 0.5 * config->simSize),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, complexInfo, config, rng
    );

    chainStartIdx = nActiveParticles->h_Current[0];
    fillParticlesStraightLine(
        &chainMembers,
        make_float3(0.5 * config->simSize - 1.8570869001109949, 0.5 * config->simSize + 2.8257686272096096, 0.5 * config->simSize - 16.748319527406373 - 2.0),
        make_float3(0.0, -1.5, 0.0),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );
    chainEndIdx = nActiveParticles->h_Current[0];
    linkParticlesSerially<Particle>(
        chainStartIdx,
        chainEndIdx,
        1,
        particles->h_Current, complexificationInfo, config, rng
    );


    instantiateComplex(
        1,
        make_float3(0.5 * config->simSize, 0.5 * config->simSize, 0.3 * config->simSize),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, complexInfo, config, rng
    );

    //chainStartIdx = nActiveParticles->h_Current[0];
    //fillParticlesStraightLine(
    //    &chainMembers,
    //    make_float3(0.5 * config->simSize - 1.8570869001109949, 0.5 * config->simSize + 2.8257686272096096, 0.3 * config->simSize - 16.748319527406373 - 2.0),
    //    make_float3(0.0, -1.5, 0.0),
    //    particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    //);
    //chainEndIdx = nActiveParticles->h_Current[0];
    //linkParticlesSerially<Particle>(
    //    chainStartIdx,
    //    chainEndIdx,
    //    1,
    //    particles->h_Current, complexificationInfo, config, rng
    //);


    //instantiateComplex(
    //    1,
    //    make_float3(0.5 * config->simSize, 0.5 * config->simSize, 0.7 * config->simSize),
    //    particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, complexInfo, config, rng
    //);

    //chainStartIdx = nActiveParticles->h_Current[0];
    //fillParticlesStraightLine(
    //    &chainMembers,
    //    make_float3(0.5 * config->simSize - 1.8570869001109949, 0.5 * config->simSize + 2.8257686272096096, 0.7 * config->simSize - 16.748319527406373 - 2.0),
    //    make_float3(0.0, -1.5, 0.0),
    //    particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    //);
    //chainEndIdx = nActiveParticles->h_Current[0];
    //linkParticlesSerially<Particle>(
    //    chainStartIdx,
    //    chainEndIdx,
    //    1,
    //    particles->h_Current, complexificationInfo, config, rng
    //);



    /*fillParticlesSphere(
        config->numParticles * 0.23,
        PARTICLE_TYPE_DNA,
        make_float3(0.5 * config->simSize, 0.5 * config->simSize, 0.5 * config->simSize),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );*/

    /*fillParticlesSphere(
        config->numParticles * 0.1,
        PARTICLE_TYPE_LIPID,
        make_float3(0.2 * config->simSize, 0.5 * config->simSize, 0.5 * config->simSize),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );
    fillParticlesSphere(
        config->numParticles * 0.01,
        PARTICLE_TYPE_DNA,
        make_float3(0.2 * config->simSize, 0.5 * config->simSize, 0.5 * config->simSize),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );*/
    /*fillParticlesSphere(
        config->numParticles * 0.15,
        PARTICLE_TYPE_DNA,
        make_float3(0.5 * config->simSize, 0.5 * config->simSize, 0.5 * config->simSize),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );
    fillParticlesSphere(
        config->numParticles * 0.08,
        PARTICLE_TYPE_DNA,
        make_float3(0.5 * config->simSize, 0.5 * config->simSize, 0.5 * config->simSize),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );*/
    /*fillParticlesSphere(
        config->numParticles * 0.05,
        PARTICLE_TYPE_DNA,
        make_float3(0.5 * config->simSize, 0.5 * config->simSize, 0.5 * config->simSize),
        particles->h_Current, nActiveParticles->h_Current, particleTypeInfo, config, rng
    );*/

    particles->copyToDevice();

    // Set the reference particle numbers/indices
    lastActiveParticle->h_Current[0] = nActiveParticles->h_Current[0] - 1;
    nextParticleId->h_Current[0] = nActiveParticles->h_Current[0];
    nActiveParticles->copyToDevice();
    lastActiveParticle->copyToDevice();
    nextParticleId->copyToDevice();
}

void
setupMetabolicParticles(
    DoubleBuffer<MetabolicParticle>* metabolicParticles,
    SingleBuffer<int>* nActiveMetabolicParticles,
    std::unordered_map<int, ParticleTypeInfo>* particleTypeInfo,
    const Config* config,
    std::function<double()> rng
) {
    fillParticlesUniform<MetabolicParticle>(
        config->numMetabolicParticles * 0.95f,
        PARTICLE_TYPE_METABOLIC,
        metabolicParticles->h_Current, nActiveMetabolicParticles->h_Current, particleTypeInfo, config, rng
    );
    addMetabolitesByCoord(
        0,
        config->numMetabolicParticles,
        0,
        metabolicParticles->h_Current, config, rng
    );

    metabolicParticles->copyToDevice();
    nActiveMetabolicParticles->copyToDevice();
}
