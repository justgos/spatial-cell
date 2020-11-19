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

__global__ void
move(
    const int step,
    curandState* rngState,
    const Particle* curParticles,
    Particle* nextParticles,
    int stepStart_nActiveParticles,
    int* nActiveParticles,
    int* lastActiveParticle,
    unsigned int* indices,
    unsigned int* gridRanges
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= stepStart_nActiveParticles)
        return;

    Particle p = curParticles[idx];

    if (!(p.flags & PARTICLE_FLAG_ACTIVE)) {
        nextParticles[idx] = p;
        return;
    }

    float3 moveVec = make_float3(0.0f, 0.0f, 0.0f);
    float3 attractionVec = make_float3(0.0f, 0.0f, 0.0f);
    constexpr float clipImpulse = 10.0f;
    constexpr float impulseScale = 0.0001f;

    // Grid cell index of the current particle
    const int cgx = getGridIdx(p.pos.x),
        cgy = getGridIdx(p.pos.y),
        cgz = getGridIdx(p.pos.z);

    // Up direction of the current particle
    float3 up = transform_vector(make_float3(0, 1, 0), p.rot);

    bool shouldBeRemoved = false;
    float nPartners = 0.0;
    for (int gx = max(cgx - 1, 0); gx <= min(cgx + 1, d_Config.nGridCells - 1); gx++) {
        for (int gy = max(cgy - 1, 0); gy <= min(cgy + 1, d_Config.nGridCells - 1); gy++) {
            for (int gz = max(cgz - 1, 0); gz <= min(cgz + 1, d_Config.nGridCells - 1); gz++) {
                // Get the range of particle ids in this block
                const unsigned int startIdx = gridRanges[makeIdx(gx, gy, gz) * 2];
                const unsigned int endIdx = gridRanges[makeIdx(gx, gy, gz) * 2 + 1];
                for (int j = startIdx; j < endIdx; j++) {
                    if (j == idx)
                        continue;

                    const Particle tp = curParticles[j];
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

                    // Up direction of the other particle
                    float3 tup = transform_vector(make_float3(0, 1, 0), tp.rot);

                    // Rotation between the particle's up direction and the direction to the other particle
                    float4 relativePositionRotation = idx < j ? quaternionFromTo(up, normalizedDelta) : quaternionFromTo(tup, negate(normalizedDelta));
                    float relativePositionAngle = angle(relativePositionRotation);
                    // Rotation between the particles' up directions
                    //float4 orientationDifference = idx < j ? mul(tp.rot, inverse(p.rot)) : mul(p.rot, inverse(tp.rot));
                    float4 orientationDifference = idx < j ? quaternionFromTo(up, tup) : quaternionFromTo(tup, up);
                    float orientationDifferenceAngle = angle(orientationDifference);

                    float interactionAngle = PI;
                    float interactionAngleMaxDelta = 0.2f;
                    float orientationAngleDelta = fabs(fabs(orientationDifferenceAngle) - interactionAngle);
                    float relativePositionAngleDelta = fabs(fabs(relativePositionAngle) - interactionAngle);
                    if (
                        p.type == 0
                        && tp.type == 1
                        && dist <= 0.005
                        && step > 10
                        && orientationAngleDelta < interactionAngleMaxDelta
                        && relativePositionAngleDelta < interactionAngleMaxDelta
                    ) {
                        shouldBeRemoved = true;
                    }
                    if (
                        p.type == 1
                        && tp.type == 1
                        && dist <= 0.005
                        && step > 10
                        && orientationAngleDelta < interactionAngleMaxDelta
                        && relativePositionAngleDelta < interactionAngleMaxDelta
                    ) {
                        // Make sure the particle creation happens only once
                        if (idx < j) {
                            int newIdx = atomicAdd(lastActiveParticle, 1);
                            if (newIdx < d_Config.numParticles) {
                                Particle np = Particle();
                                np.pos = make_float3(
                                    p.pos.x + delta.x / 2,
                                    p.pos.y + delta.y / 2,
                                    p.pos.z + delta.z / 2
                                );
                                np.rot = random_rotation(&rngState[idx]);
                                np.velocity = make_float3(0, 0, 0);
                                np.type = 2;
                                np.flags = PARTICLE_FLAG_ACTIVE;
                                nextParticles[newIdx] = np;
                                atomicAdd(nActiveParticles, 1);
                            }
                        }
                        p.type = 3;
                    }

                    float repulsion = -fmin(1 / (pow(dist * 400.0f, 2.0f) + 1e-6f), clipImpulse) * impulseScale;
                    float attraction = 0.0f;
                    // Graph: https://www.desmos.com/calculator/wdnrfaaqps
                    if (p.type == 0 && p.type == tp.type) {
                        attraction = 0.7 * (exp(-pow(abs(dist) * 100.0f, 2.0f)) * impulseScale * 10 - fmin(1 / (pow(dist * 70.0f, 2.0f) + 1e-6f), clipImpulse) * impulseScale);
                        attractionVec.x += copysign(1.0, delta.x) * (attraction);
                        attractionVec.y += copysign(1.0, delta.y) * (attraction);
                        attractionVec.z += copysign(1.0, delta.z) * (attraction);
                    }
                    moveVec.x += copysign(1.0, delta.x) * (repulsion);
                    moveVec.y += copysign(1.0, delta.y) * (repulsion);
                    moveVec.z += copysign(1.0, delta.z) * (repulsion);
                    nPartners += 1.0;
                }
            }
        }
    }

    // Mark the particle as inactive
    if (shouldBeRemoved) {
        p.flags = p.flags ^ PARTICLE_FLAG_ACTIVE;
        atomicAdd(nActiveParticles, -1);
    }

    // Prevent attraction overkill for large aggregations
    nPartners = pow(fmax(nPartners, 1.0f), 0.5f);
    attractionVec.x /= nPartners;
    attractionVec.y /= nPartners;
    attractionVec.z /= nPartners;
    moveVec.x += attractionVec.x;
    moveVec.y += attractionVec.y;
    moveVec.z += attractionVec.z;

    // Brownian motion
    moveVec.x += (curand_normal(&rngState[idx]) - 0.0) * d_Config.movementNoiseScale;
    moveVec.y += (curand_normal(&rngState[idx]) - 0.0) * d_Config.movementNoiseScale;
    moveVec.z += (curand_normal(&rngState[idx]) - 0.0) * d_Config.movementNoiseScale;

    // Apply the velocity changes
    p.velocity.x *= d_Config.velocityDecay;
    p.velocity.y *= d_Config.velocityDecay;
    p.velocity.z *= d_Config.velocityDecay;
    p.velocity.x += moveVec.x;
    p.velocity.y += moveVec.y;
    p.velocity.z += moveVec.z;

    // Brownian rotation
    p.rot = lepr(p.rot, random_rotation(&rngState[idx]), d_Config.rotationNoiseScale);

    // Move the particle
    p.pos.x = fmin(fmax(p.pos.x + p.velocity.x, 0.0f), d_Config.simSize);
    p.pos.y = fmin(fmax(p.pos.y + p.velocity.y, 0.0f), d_Config.simSize);
    p.pos.z = fmin(fmax(p.pos.z + p.velocity.z, 0.0f), d_Config.simSize);

    nextParticles[idx] = p;
}
