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
    int* nextParticleId,
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
    float3 up = transform_vector(VECTOR_UP, p.rot);

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
                    float3 tup = transform_vector(VECTOR_UP, tp.rot);

                    // Rotation between the particle's up direction and the direction to the other particle
                    float4 relativePositionRotation = p.id < tp.id ? quaternionFromTo(up, normalizedDelta) : quaternionFromTo(tup, negate(normalizedDelta));
                    float relativePositionAngle = angle(relativePositionRotation);
                    // Rotation between the particles' up directions
                    //float4 orientationDifference = p.id < tp.id ? mul(tp.rot, inverse(p.rot)) : mul(p.rot, inverse(tp.rot));
                    float4 orientationDifference = p.id < tp.id ? quaternionFromTo(up, tup) : quaternionFromTo(tup, up);
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
                        if (p.id < tp.id) {
                            int newIdx = atomicAdd(lastActiveParticle, 1);
                            if (newIdx < d_Config.numParticles) {
                                int newId = atomicAdd(nextParticleId, 1);
                                Particle np = Particle();
                                np.id = newId;
                                np.type = 2;
                                np.flags = PARTICLE_FLAG_ACTIVE;
                                np.pos = make_float3(
                                    p.pos.x + delta.x / 2,
                                    p.pos.y + delta.y / 2,
                                    p.pos.z + delta.z / 2
                                );
                                np.rot = random_rotation(&rngState[idx]);
                                np.velocity = VECTOR_ZERO;
                                np.nActiveInteractions = 0;
                                nextParticles[newIdx] = np;
                                atomicAdd(nActiveParticles, 1);
                            }
                        }
                        p.type = 3;
                    }

                    float repulsion = -fmin(1 / (pow(dist * 400.0f, 2.0f) + 1e-6f), clipImpulse) * impulseScale;
                    float attraction = 0.0f;

                    for (int k = 0; k < p.nActiveInteractions; k++) {
                        ParticleInteraction interaction = p.interactions[k];
                        if (interaction.partnerId == tp.id) {
                            attraction = 2.7 * (exp(-pow(abs(dist) * 100.0f, 2.0f)) * impulseScale * 10 - fmin(1 / (pow(dist * 70.0f, 2.0f) + 1e-6f), clipImpulse) * impulseScale);
                            attractionVec.x += copysign(1.0, delta.x) * (attraction);
                            attractionVec.y += copysign(1.0, delta.y) * (attraction);
                            attractionVec.z += copysign(1.0, delta.z) * (attraction);
                        }
                    }

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
    p.rot = slerp(p.rot, random_rotation(&rngState[idx]), d_Config.rotationNoiseScale);

    // Move the particle
    p.pos.x = fmin(fmax(p.pos.x + p.velocity.x, 0.0f), d_Config.simSize);
    p.pos.y = fmin(fmax(p.pos.y + p.velocity.y, 0.0f), d_Config.simSize);
    p.pos.z = fmin(fmax(p.pos.z + p.velocity.z, 0.0f), d_Config.simSize);

    nextParticles[idx] = p;
}


__global__ void
relax(
    const int step,
    curandState* rngState,
    const Particle* curParticles,
    Particle* nextParticles,
    int stepStart_nActiveParticles,
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
    //float4 rotQuat = QUATERNION_IDENTITY;

    // Grid cell index of the current particle
    const int cgx = getGridIdx(p.pos.x),
        cgy = getGridIdx(p.pos.y),
        cgz = getGridIdx(p.pos.z);

    // Up direction of the current particle
    float3 up = transform_vector(VECTOR_UP, p.rot);

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

                    for (int k = 0; k < p.nActiveInteractions; k++) {
                        ParticleInteraction interaction = p.interactions[k];
                        if (interaction.partnerId == tp.id) {
                            float interactionDistance = 0.005;
                            float distRatio = (interactionDistance - dist) / (dist + 1e-6);
                            constexpr float collisionRelaxationSpeed = 0.25f;
                            /*moveVec.x += -delta.x * distRatio * collisionRelaxationSpeed;
                            moveVec.y += -delta.y * distRatio * collisionRelaxationSpeed;
                            moveVec.z += -delta.z * distRatio * collisionRelaxationSpeed;*/

                            // Align relative orientation
                            float4 targetRelativeOrientation = mul(quaternion(VECTOR_UP, -PI / 6), quaternion(VECTOR_RIGHT, PI / 6));
                            float4 relativeOrientationDelta = p.id < tp.id 
                                ? mul(tp.rot, inverse(targetRelativeOrientation))
                                : mul(tp.rot, targetRelativeOrientation);
                            constexpr float relativeOrientationRelaxationSpeed = 0.2f;
                            p.rot = slerp(p.rot, relativeOrientationDelta, relativeOrientationRelaxationSpeed);

                            // Align relative position
                            //float4 relativePositionRotation = p.id < tp.id ? quaternionFromTo(up, normalizedDelta) : quaternionFromTo(tup, negate(normalizedDelta));
                            float4 targetRelativePositionRotation = quaternion(VECTOR_FORWARD, -PI / 6);
                            /*float4 deltaRelativePositionOrientation = p.id < tp.id 
                                ? mul(targetRelativePositionRotation, inverse(quaternionFromTo(normalizedDelta, up)))
                                : mul(targetRelativePositionRotation, inverse(quaternionFromTo(negate(normalizedDelta), up)));*/
                            /*float4 deltaRelativePositionOrientation = p.id < tp.id
                                ? quaternionFromTo(up, normalizedDelta)
                                : quaternionFromTo(up, negate(normalizedDelta));*/
                            //p.rot = lerp(p.rot, mul(p.rot, deltaRelativePositionOrientation), relativeOrientationRelaxationSpeed);
                            //targetRelativePositionRotation = p.id < tp.id ? mul(targetRelativePositionRotation, p.rot) : mul(targetRelativePositionRotation, tp.rot);
                            constexpr float relativePositionRelaxationSpeed = 0.2f;
                            float3 relaxedRelativePosition = p.id < tp.id 
                                ? negate(transform_vector(VECTOR_UP, mul(tp.rot, mul(inverse(targetRelativeOrientation), targetRelativePositionRotation))))
                                : transform_vector(VECTOR_UP, mul(tp.rot, targetRelativePositionRotation));
                            /*float3 relaxedRelativePosition = p.id < tp.id
                                ? make_float3(0, -1, 0)
                                : VECTOR_UP;*/
                            relaxedRelativePosition.x *= interactionDistance;
                            relaxedRelativePosition.y *= interactionDistance;
                            relaxedRelativePosition.z *= interactionDistance;
                            relaxedRelativePosition.x += tp.pos.x;
                            relaxedRelativePosition.y += tp.pos.y;
                            relaxedRelativePosition.z += tp.pos.z;
                            moveVec.x += (relaxedRelativePosition.x - p.pos.x) * relativePositionRelaxationSpeed;
                            moveVec.y += (relaxedRelativePosition.y - p.pos.y) * relativePositionRelaxationSpeed;
                            moveVec.z += (relaxedRelativePosition.z - p.pos.z) * relativePositionRelaxationSpeed;

                            p.debugVector.x = (relaxedRelativePosition.x - p.pos.x) * (1.0 - relativePositionRelaxationSpeed);
                            p.debugVector.y = (relaxedRelativePosition.y - p.pos.y) * (1.0 - relativePositionRelaxationSpeed);
                            p.debugVector.z = (relaxedRelativePosition.z - p.pos.z) * (1.0 - relativePositionRelaxationSpeed);
                        }
                    }
                }
            }
        }
    }

    // Move the particle
    p.pos.x = fmin(fmax(p.pos.x + moveVec.x, 0.0f), d_Config.simSize);
    p.pos.y = fmin(fmax(p.pos.y + moveVec.y, 0.0f), d_Config.simSize);
    p.pos.z = fmin(fmax(p.pos.z + moveVec.z, 0.0f), d_Config.simSize);

    //p.rot = mul(p.rot, rotQuat);

    nextParticles[idx] = p;
}
