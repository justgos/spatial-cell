#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <crt/math_functions.h>

#include "types.cuh"
#include "constants.cuh"
#include "macros.cuh"
#include "math.cuh"
#include "grid.cuh"
#include "memory.cuh"
#include "interactions.cuh"


__global__ void
moveMetabolicParticles(
    const int step,
    curandState* metabolicParticleRngState,
    const MetabolicParticle* curMetabolicParticles,
    MetabolicParticle* nextMetabolicParticles,
    int stepStart_nActiveMetabolicParticles,
    unsigned int* metabolicParticleGridRanges
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= stepStart_nActiveMetabolicParticles)
        return;

    MetabolicParticle p = curMetabolicParticles[idx];

    if (!(p.flags & PARTICLE_FLAG_ACTIVE)) {
        nextMetabolicParticles[idx] = p;
        return;
    }

    float3 moveVec = make_float3(0.0f, 0.0f, 0.0f);

    // Brownian motion
    moveVec.x += (curand_normal(&metabolicParticleRngState[idx]) - 0.0) * d_Config.metaboliteMovementNoiseScale;
    moveVec.y += (curand_normal(&metabolicParticleRngState[idx]) - 0.0) * d_Config.metaboliteMovementNoiseScale;
    moveVec.z += (curand_normal(&metabolicParticleRngState[idx]) - 0.0) * d_Config.metaboliteMovementNoiseScale;

    // Apply the velocity changes
    p.velocity.x *= d_Config.velocityDecay;
    p.velocity.y *= d_Config.velocityDecay;
    p.velocity.z *= d_Config.velocityDecay;
    p.velocity.x += moveVec.x;
    p.velocity.y += moveVec.y;
    p.velocity.z += moveVec.z;

    // Brownian rotation
    p.rot = slerp(p.rot, random_rotation(&metabolicParticleRngState[idx]), d_Config.rotationNoiseScale);
    
    // Move the particle
    p.pos.x = fmin(fmax(p.pos.x + p.velocity.x, 0.0f), d_Config.simSize);
    p.pos.y = fmin(fmax(p.pos.y + p.velocity.y, 0.0f), d_Config.simSize);
    p.pos.z = fmin(fmax(p.pos.z + p.velocity.z, 0.0f), d_Config.simSize);

    nextMetabolicParticles[idx] = p;
}

__global__ void
relaxMetabolicParticles(
    const int step,
    curandState* metabolicParticleRngState,
    const MetabolicParticle* curMetabolicParticles,
    MetabolicParticle* nextMetabolicParticles,
    int stepStart_nActiveMetabolicParticles,
    unsigned int* metabolicParticleGridRanges,
    const Particle* particles,
    unsigned int* gridRanges
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= stepStart_nActiveMetabolicParticles)
        return;

    MetabolicParticle p = curMetabolicParticles[idx];

    if (!(p.flags & PARTICLE_FLAG_ACTIVE)) {
        nextMetabolicParticles[idx] = p;
        return;
    }

    float3 moveVec = make_float3(0.0f, 0.0f, 0.0f);

    // Grid cell index of the current particle
    const int cgx = getGridIdx(p.pos.x),
        cgy = getGridIdx(p.pos.y),
        cgz = getGridIdx(p.pos.z);

    // Up direction of the current particle
    float3 up = transform_vector(VECTOR_UP, p.rot);

    p.debugVector.x = 0;
    p.debugVector.y = 0;
    p.debugVector.z = 0;

    for (int gx = max(cgx - 1, 0); gx <= min(cgx + 1, d_Config.nGridCells - 1); gx++) {
        for (int gy = max(cgy - 1, 0); gy <= min(cgy + 1, d_Config.nGridCells - 1); gy++) {
            for (int gz = max(cgz - 1, 0); gz <= min(cgz + 1, d_Config.nGridCells - 1); gz++) {
                // Get the range of other metabolic particle ids in this block
                const unsigned int metabolicParticleStartIdx = metabolicParticleGridRanges[makeIdx(gx, gy, gz) * 2];
                const unsigned int metabolicParticleEndIdx = metabolicParticleGridRanges[makeIdx(gx, gy, gz) * 2 + 1];
                for (int j = metabolicParticleStartIdx; j < metabolicParticleEndIdx; j++) {
                    if (j == idx)
                        continue;

                    const MetabolicParticle tp = curMetabolicParticles[j];
                    if (!(tp.flags & PARTICLE_FLAG_ACTIVE))
                        continue;

                    float3 delta = make_float3(
                        tp.pos.x - p.pos.x,
                        tp.pos.y - p.pos.y,
                        tp.pos.z - p.pos.z
                    );
                    // Skip particles beyong the maximum interaction distance
                    if (fabs(delta.x) > d_Config.interactionDistance
                        || fabs(delta.y) > d_Config.interactionDistance
                        || fabs(delta.y) > d_Config.interactionDistance)
                        continue;

                    float3 normalizedDelta = normalized(delta);
                    float dist = norm(delta);

                    float collisionDist = 0.005;

                    // Up direction of the other particle
                    float3 tup = transform_vector(VECTOR_UP, tp.rot);

                    if (dist <= collisionDist) {
                        float distRatio = (collisionDist - dist) / (dist + 1e-6);
                        constexpr float collisionRelaxationSpeed = 0.25f;
                        moveVec.x += -delta.x * distRatio * collisionRelaxationSpeed;
                        moveVec.y += -delta.y * distRatio * collisionRelaxationSpeed;
                        moveVec.z += -delta.z * distRatio * collisionRelaxationSpeed;
                    }
                }

                // Get the range of plain particle ids in this block
                const unsigned int startIdx = gridRanges[makeIdx(gx, gy, gz) * 2];
                const unsigned int endIdx = gridRanges[makeIdx(gx, gy, gz) * 2 + 1];
                for (int j = startIdx; j < endIdx; j++) {
                    const Particle tp = particles[j];
                    // Repulsion only from lipids

                    /*
                    * TODO: repel the _lipids_ from the metabolic particles,
                    *       so that vesicles full of small molecules would preserve their volume
                    */

                    if (
                        !(tp.flags & PARTICLE_FLAG_ACTIVE)
                        || tp.type != PARTICLE_TYPE_LIPID
                    )
                        continue;

                    float3 delta = make_float3(
                        tp.pos.x - p.pos.x,
                        tp.pos.y - p.pos.y,
                        tp.pos.z - p.pos.z
                    );
                    // Skip particles beyong the maximum interaction distance
                    if (fabs(delta.x) > d_Config.interactionDistance
                        || fabs(delta.y) > d_Config.interactionDistance
                        || fabs(delta.y) > d_Config.interactionDistance)
                        continue;

                    float3 normalizedDelta = normalized(delta);
                    float dist = norm(delta);

                    float collisionDist = 0.006;

                    // Up direction of the other particle
                    float3 tup = transform_vector(VECTOR_UP, tp.rot);

                    if (dist <= collisionDist) {
                        float distRatio = (collisionDist - dist) / (dist + 1e-6);
                        constexpr float collisionRelaxationSpeed = 0.25f;
                        moveVec.x += -delta.x * distRatio * collisionRelaxationSpeed;
                        moveVec.y += -delta.y * distRatio * collisionRelaxationSpeed;
                        moveVec.z += -delta.z * distRatio * collisionRelaxationSpeed;
                    }
                }
            }
        }
    }

    // Move the particle
    p.pos.x = fmin(fmax(p.pos.x + moveVec.x, 0.0f), d_Config.simSize);
    p.pos.y = fmin(fmax(p.pos.y + moveVec.y, 0.0f), d_Config.simSize);
    p.pos.z = fmin(fmax(p.pos.z + moveVec.z, 0.0f), d_Config.simSize);

    nextMetabolicParticles[idx] = p;
}

__global__ void
diffuseMetabolites(
    const int step,
    curandState* metabolicParticleRngState,
    const MetabolicParticle* curMetabolicParticles,
    MetabolicParticle* nextMetabolicParticles,
    int stepStart_nActiveMetabolicParticles,
    unsigned int* metabolicParticleGridRanges
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= stepStart_nActiveMetabolicParticles)
        return;

    MetabolicParticle p = curMetabolicParticles[idx];

    if (!(p.flags & PARTICLE_FLAG_ACTIVE)) {
        nextMetabolicParticles[idx] = p;
        return;
    }

    float deltaMetabolites[NUM_METABOLITES] = { 0 };
    //memset(deltaMetabolites, 0, NUM_METABOLITES * sizeof(float));

    // Grid cell index of the current particle
    const int cgx = getGridIdx(p.pos.x),
        cgy = getGridIdx(p.pos.y),
        cgz = getGridIdx(p.pos.z);

    for (int gx = max(cgx - 1, 0); gx <= min(cgx + 1, d_Config.nGridCells - 1); gx++) {
        for (int gy = max(cgy - 1, 0); gy <= min(cgy + 1, d_Config.nGridCells - 1); gy++) {
            for (int gz = max(cgz - 1, 0); gz <= min(cgz + 1, d_Config.nGridCells - 1); gz++) {
                // Get the range of other metabolic particle ids in this block
                const unsigned int metabolicParticleStartIdx = metabolicParticleGridRanges[makeIdx(gx, gy, gz) * 2];
                const unsigned int metabolicParticleEndIdx = metabolicParticleGridRanges[makeIdx(gx, gy, gz) * 2 + 1];
                for (int j = metabolicParticleStartIdx; j < metabolicParticleEndIdx; j++) {
                    if (j == idx)
                        continue;

                    const MetabolicParticle tp = curMetabolicParticles[j];
                    if (!(tp.flags & PARTICLE_FLAG_ACTIVE))
                        continue;

                    float3 delta = make_float3(
                        tp.pos.x - p.pos.x,
                        tp.pos.y - p.pos.y,
                        tp.pos.z - p.pos.z
                    );
                    // Skip particles beyong the maximum interaction distance
                    if (fabs(delta.x) > d_Config.interactionDistance
                        || fabs(delta.y) > d_Config.interactionDistance
                        || fabs(delta.y) > d_Config.interactionDistance)
                        continue;

                    float3 normalizedDelta = normalized(delta);
                    float dist = norm(delta);

                    const float diffusionDistance = 0.008;
                    if (dist <= diffusionDistance) {
                        for (int k = 0; k < NUM_METABOLITES; k++) {
                            float diff = tp.metabolites[k] - p.metabolites[k];
                            //// TODO: shuffle around single metabolites using `scramble_float`
                            //if (fabs(diff) >= 2.0) {
                            //    //
                            //}
                            if (fabs(diff) > 0.001) {
                                float diffusionSpeed = 0.1f;
                                deltaMetabolites[k] += diff * diffusionSpeed;
                            }
                        }
                    }
                }
            }
        }
    }

    for (int k = 0; k < NUM_METABOLITES; k++)
        p.metabolites[k] += deltaMetabolites[k];

    nextMetabolicParticles[idx] = p;
}
