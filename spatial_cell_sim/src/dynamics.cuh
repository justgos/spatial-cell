#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <crt/math_functions.h>

#include "../deps/cudaNoise/cudaNoise.cuh"

#include "types.cuh"
#include "constants.cuh"
#include "macros.cuh"
#include "math.cuh"
#include "grid.cuh"
#include "memory.cuh"
#include "interactions.cuh"

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

    p.debugVector.x = 0;
    p.debugVector.y = 0;
    p.debugVector.z = 0;

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
                    // Skip particles beyond the maximum interaction distance
                    if (fabs(delta.x) > d_Config.maxInteractionDistance
                        || fabs(delta.y) > d_Config.maxInteractionDistance
                        || fabs(delta.z) > d_Config.maxInteractionDistance)
                        continue;

                    float3 normalizedDelta = normalize(delta);
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
                    float orientationAngleDelta = fabs(orientationDifferenceAngle - interactionAngle);
                    float relativePositionAngleDelta = fabs(relativePositionAngle - interactionAngle);
                    /*if (
                        p.type == 0
                        && tp.type == 1
                        && dist <= p.radius + tp.radius
                        && step > 10
                        && orientationAngleDelta < interactionAngleMaxDelta
                        && relativePositionAngleDelta < interactionAngleMaxDelta
                    ) {
                        shouldBeRemoved = true;
                    }*/
                    if (
                        p.type == 1
                        && tp.type == 1
                        && dist <= p.radius + tp.radius
                        && step > 10
                        && orientationAngleDelta < interactionAngleMaxDelta
                        && relativePositionAngleDelta < interactionAngleMaxDelta
                    ) {
                        // Make sure the particle creation happens only once
                        if (p.id < tp.id) {
                            int newIdx = atomicAdd(lastActiveParticle, 1);
                            if (newIdx < d_Config.numParticles) {
                                int newId = atomicAdd(nextParticleId, 1);
                                Particle np = Particle(
                                    newId,
                                    2,
                                    0,
                                    0.001,
                                    make_float3(
                                        p.pos.x + delta.x / 2,
                                        p.pos.y + delta.y / 2,
                                        p.pos.z + delta.z / 2
                                    ),
                                    random_rotation(&rngState[idx])
                                );
                                nextParticles[newIdx] = np;
                                atomicAdd(nActiveParticles, 1);
                            }
                        }
                        p.radius = 0.002;
                        p.type = 3;
                    }
                }
            }
        }
    }

    // Mark the particle as inactive
    if (shouldBeRemoved) {
        p.flags = p.flags & ~PARTICLE_FLAG_ACTIVE;
        atomicAdd(nActiveParticles, -1);
    }

    nextParticles[idx] = p;
}

template <typename T>
__global__ void
brownianMovementAndRotation(
    const int step,
    curandState* rngState,
    T* particles,
    const int stepStart_nActiveParticles,
    const float movementNoiseScale
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= stepStart_nActiveParticles)
        return;

    T p = particles[idx];

    if (!(p.flags & PARTICLE_FLAG_ACTIVE)) {
        return;
    }

    // Brownian motion
    p.velocity = add(
        mul(p.velocity, d_Config.velocityDecay),
        mul(
            transform_vector(VECTOR_UP, random_rotation(&rngState[idx])),
            fabs(curand_normal(&rngState[idx])) * movementNoiseScale
        )
    );
    //float3 noisePos = p.pos + step * make_float3(0.1, 0.1, 0.1) * 100.0;
    //float noiseScale = 10.0;
    ///*p.velocity = p.velocity * d_Config.velocityDecay + make_float3(
    //    cudaNoise::simplexNoise(noisePos, noiseScale, 42),
    //    cudaNoise::simplexNoise(noisePos, noiseScale, 84),
    //    cudaNoise::simplexNoise(noisePos, noiseScale, 126)
    //) * pow(clamp(fabs(cudaNoise::simplexNoise(noisePos, noiseScale, 43)), 0.0, 0.9999), 2.0f) * 4.0f * movementNoiseScale;*/
    //p.velocity = add(
    //    mul(p.velocity, d_Config.velocityDecay),
    //    mul(
    //        transform_vector(VECTOR_UP, random_rotation(noisePos, noiseScale, 42)),
    //        clamp(fabs(cudaNoise::simplexNoise(noisePos, noiseScale, 43)), 0.0, 1.0) * movementNoiseScale
    //    )
    //);

    // Brownian rotation
    p.angularVelocity = slerp(
        slerp(p.angularVelocity, QUATERNION_IDENTITY, d_Config.angularVelocityDecay),
        random_rotation(&rngState[idx]),
        d_Config.rotationNoiseScale
    );
    
    particles[idx] = p;
}

template <typename T>
__global__ void
applyVelocities(
    const int step,
    curandState* rngState,
    T* particles,
    int stepStart_nActiveParticles,
    float stepFraction
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= stepStart_nActiveParticles)
        return;

    T p = particles[idx];

    if (!(p.flags & PARTICLE_FLAG_ACTIVE)) {
        return;
    }

    // Move the particle
    p.pos = clamp(
        add(
            p.pos,
            mul(p.velocity, stepFraction)
        ),
        0, d_Config.simSize
    );

    // Rotate the particle
    p.rot = mul(
        p.rot,
        slerp(QUATERNION_IDENTITY, p.angularVelocity, stepFraction)
    );

    particles[idx] = p;
}


/*
* Noise applied to interacted partners should be aligned
*/
__global__ void
coordinateNoise(
    const int step,
    curandState* rngState,
    const Particle* curParticles,
    Particle* nextParticles,
    int stepStart_nActiveParticles,
    unsigned int* gridRanges
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= stepStart_nActiveParticles)
        return;

    Particle p = curParticles[idx];

    if (
        !(p.flags & PARTICLE_FLAG_ACTIVE)
        || p.nActiveInteractions < 1
    ) {
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
                    // Skip particles beyond the maximum interaction distance
                    if (fabs(delta.x) > d_Config.maxInteractionDistance
                        || fabs(delta.y) > d_Config.maxInteractionDistance
                        || fabs(delta.z) > d_Config.maxInteractionDistance)
                        continue;

                    float3 normalizedDelta = normalize(delta);

                    for (int k = 0; k < p.nActiveInteractions; k++) {
                        ParticleInteraction interaction = p.interactions[k];
                        if (interaction.partnerId == tp.id) {
                            // Noise should act on the interacting partners as if they were one
                            p.velocity = lerp(p.velocity, tp.velocity, 0.49f / (p.nActiveInteractions + tp.nActiveInteractions));

                            p.velocity = p.velocity - -normalizedDelta * min(dot(p.velocity, -normalizedDelta), 0.0) * 0.5;

                            p.angularVelocity = slerp(
                                p.angularVelocity,
                                tp.angularVelocity,
                                0.49f / (p.nActiveInteractions + tp.nActiveInteractions)
                            );
                        }
                    }
                }
            }
        }
    }

    nextParticles[idx] = p;
}


__global__ void
relax(
    const int step,
    curandState* rngState,
    const Particle* curParticles,
    Particle* nextParticles,
    int stepStart_nActiveParticles,
    unsigned int* gridRanges,
    ParticleInteractionInfo *flatComplexificationInfo
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
                    // Skip particles beyond the maximum interaction distance
                    if (fabs(delta.x) > d_Config.maxInteractionDistance
                        || fabs(delta.y) > d_Config.maxInteractionDistance
                        || fabs(delta.z) > d_Config.maxInteractionDistance)
                        continue;

                    float3 normalizedDelta = normalize(delta);
                    float dist = norm(delta);

                    float interactionDistance = p.radius + tp.radius;

                    // Up direction of the other particle
                    float3 tup = transform_vector(VECTOR_UP, tp.rot);                    

                    /*
                    * TODO: interactions should alter the particle's velocity, not just position
                    */

                    if (p.type == PARTICLE_TYPE_LIPID && p.type == tp.type && dist <= p.radius + tp.radius + 0.8 * p.radius) {
                        float deltaInteractionDist = -(interactionDistance - dist);
                        constexpr float distanceRelaxationSpeed = 0.15f;
                        moveVec += normalizedDelta * (deltaInteractionDist * distanceRelaxationSpeed);

                        float interactionAngleMaxDelta = PI / 4;
                        float interactionOrientationAngle = 0;
                        float interactionRelativePositionsAngle = PI / 2;
                        // Rotation between the particles' up directions
                            //float4 orientationDifference = p.id < tp.id ? mul(tp.rot, inverse(p.rot)) : mul(p.rot, inverse(tp.rot));
                        float4 orientationDifference = p.id < tp.id ? quaternionFromTo(up, tup) : quaternionFromTo(tup, up);
                        float orientationDifferenceAngle = angle(orientationDifference);
                        float orientationAngleDelta = fabs(orientationDifferenceAngle - interactionOrientationAngle);

                        bool correctlyOrientedAndPositioned = false;

                        if (orientationAngleDelta < interactionAngleMaxDelta) {
                            // Rotation between the particle's up direction and the direction to the other particle
                            float4 relativePositionRotation = p.id < tp.id ? quaternionFromTo(up, normalizedDelta) : quaternionFromTo(tup, negate(normalizedDelta));
                            float relativePositionAngle = angle(relativePositionRotation);
                            float relativePositionAngleDelta = fabs(relativePositionAngle - interactionRelativePositionsAngle);

                            // If the particles are aligned well enough, strengthen the alignment futher
                            if (relativePositionAngleDelta < interactionAngleMaxDelta) {
                                correctlyOrientedAndPositioned = true;
                                /*
                                * TODO: mix discontinuous noise with a spatially smooth one for aggregated particles
                                */

                                // Lipids that are part of the membrane should experience
                                // less movement noise perpendicular to the membrane surface
                                /*p.velocity = add(
                                    p.velocity,
                                    mul(
                                        negate(mul(up, dot(p.velocity, up))),
                                        0.9
                                    )
                                );*/
                                // ..or leave mostly the vertical movement
                                //p.velocity = p.velocity * 0.8 + 0.2 * (up * dot(p.velocity, up));

                                // Reduce the noise coming from the direction of the partner (the opposite-facing component), 
                                // as there're less noise water molecules there
                                p.velocity = p.velocity - -normalizedDelta * min(dot(p.velocity, -normalizedDelta), 0.0) * 0.5;

                                //p.velocity = mul(p.velocity, 0.9);
                                // and less angular noise
                                p.angularVelocity = slerp(p.angularVelocity, QUATERNION_IDENTITY, 0.9);

                                // Align relative orientation
                                float4 targetRelativeOrientationDelta = quaternion(VECTOR_RIGHT, 0);
                                constexpr float relativeOrientationRelaxationSpeed = 0.05f;
                                float targetRelativePositionAngle = PI / 2;
                                float targetRelativePositionAngleFlex = PI / 8;

                                p.rot = slerp(
                                    p.rot,
                                    //mul(quaternionFromTo(up, dot(up, tup) > 0 ? tup : negate(tup)), tp.rot),
                                    mul(quaternionFromTo(up, tup), p.rot),
                                    //dot(up, tup) > 0 ? tp.rot : mul(tp.rot, quaternion(VECTOR_RIGHT, PI)),
                                    //getTargetRelativeOrientation(p, tp, targetRelativeOrientationDelta),
                                    relativeOrientationRelaxationSpeed
                                );

                                p.rot = slerp(
                                    p.rot,
                                    mul(
                                        quaternionFromTo(
                                            up,
                                            transform_vector(
                                                normalizedDelta,
                                                quaternion(
                                                    normalize(cross(normalizedDelta, up)),
                                                    targetRelativePositionAngle
                                                )
                                            )
                                        ),
                                        p.rot
                                    ),
                                    //mul(quaternion(cross(up, normalizedDelta), targetRelativePositionAngle - angle(quaternionFromTo(up, normalizedDelta))), p.rot),
                                    //getTargetRelativeOrientation(p, tp, targetRelativeOrientationDelta),
                                    relativeOrientationRelaxationSpeed
                                );

                                //// Decrease the movement noise for each interaction
                                //p.velocity = mul(p.velocity, 0.5);

                                // Align relative position
                                constexpr float relativePositionRelaxationSpeed = 0.05f;
                                //float4 targetRelativePositionRotation = quaternion(cross(tup, negate(normalizedDelta)), targetRelativePositionAngle);
                                float3 relativePositionRelaxationAxis = normalize(cross(tup, -(normalizedDelta)));
                                float currentRelativePositionAngle = angle(tup, -(normalizedDelta));
                                /*float3 negDelta = negate(normalizedDelta);
                                float sinAngle = sin(currentRelativePositionAngle * 0.5f);
                                float cosAngle = cos(currentRelativePositionAngle * 0.5f);
                                float4 qt4 = quaternion(cross(normalize(tup), negate(normalizedDelta)), currentRelativePositionAngle);
                                float4 qt3 = quaternion(normalize(relativePositionRelaxationAxis), currentRelativePositionAngle);
                                float4 qt2 = quaternion(relativePositionRelaxationAxis, currentRelativePositionAngle);
                                float4 qt = quaternionFromTo(tup, negate(normalizedDelta));
                                float3 tv = transform_vector(tup, qt);*/
                                float3 relaxedRelativePosition = transform_vector(
                                    tup,
                                    quaternion(
                                        relativePositionRelaxationAxis,
                                        min(
                                            max(
                                                currentRelativePositionAngle,
                                                targetRelativePositionAngle - targetRelativePositionAngleFlex
                                            ),
                                            targetRelativePositionAngle + targetRelativePositionAngleFlex
                                        )
                                    )
                                );
                                relaxedRelativePosition *= interactionDistance;
                                relaxedRelativePosition += tp.pos;
                                moveVec += (relaxedRelativePosition - p.pos) * relativePositionRelaxationSpeed;

                                /*p.debugVector.x = (relaxedRelativePosition.x - p.pos.x) * (1.0 - relativePositionRelaxationSpeed);
                                p.debugVector.y = (relaxedRelativePosition.y - p.pos.y) * (1.0 - relativePositionRelaxationSpeed);
                                p.debugVector.z = (relaxedRelativePosition.z - p.pos.z) * (1.0 - relativePositionRelaxationSpeed);*/
                            }
                        }

                        // Kinda.. membrane fusion?
                        if(!correctlyOrientedAndPositioned) {
                            constexpr float verticalIrregularityRelaxationSpeed = 0.05f;
                            //moveVec += normalizedDelta * dist * verticalIrregularityRelaxationSpeed;
                            moveVec = add(
                                moveVec,
                                mul(
                                    dot(up, normalizedDelta) > 0 ? up : negate(up),
                                    dist * verticalIrregularityRelaxationSpeed
                                )
                            );
                        }
                    }

                    bool interactionPartners = false;

                    for (int k = 0; k < p.nActiveInteractions; k++) {
                        ParticleInteraction interaction = p.interactions[k];
                        if (interaction.partnerId == tp.id) {
                            interactionPartners = true;

                            if (interaction.type >= 10) {
                                ParticleInteractionInfo pii = flatComplexificationInfo[interaction.type];

                                // Noise should act on the interacting partners as if they were one
                                // TODO: broadcast the noise direction faster to reduce the complex's wobbling
                                //p.velocity = lerp(p.velocity, tp.velocity, 0.99f / (p.nActiveInteractions + tp.nActiveInteractions));
                                // Reduce the noise coming from the direction of the partner (the opposite-facing component), 
                                // as there're less noise water molecules there
                                //p.velocity = p.velocity - -normalizedDelta * min(dot(p.velocity, -normalizedDelta), 0.0) * 0.2;

                                // Align relative orientation
                                constexpr float relativeOrientationRelaxationSpeed = 0.1f;
                                p.rot = slerp(
                                    p.rot,
                                    getTargetRelativeOrientation(p, tp, pii.relativeOrientation),
                                    relativeOrientationRelaxationSpeed
                                );

                                // Align relative position
                                //constexpr float relativePositionRelaxationSpeed = 0.15f;
                                float3 relaxedRelativePosition = interactionParticipantOrder(p, tp)
                                    ? transform_vector(-pii.relativePosition, mul(inverse(pii.relativeOrientation), tp.rot))
                                    : transform_vector(pii.relativePosition, tp.rot);
                                relaxedRelativePosition += tp.pos;
                                /*getRelaxedRelativePosition(
                                    p,
                                    tp,
                                    pii.relativeOrientation,
                                    pii.relativePosition,
                                    pii.distance
                                );*/
                                float3 relativePositionDelta = relaxedRelativePosition - p.pos;
                                //moveVec += relativePositionDelta * relativePositionRelaxationSpeed;
                                moveVec += relativePositionDelta * 0.99f / (p.nActiveInteractions + tp.nActiveInteractions);
                            }

                            // Interaction testing code
                            if (interaction.type == 0) {
                                float distRatio = (interactionDistance - dist) / (dist + 1e-6);
                                constexpr float collisionRelaxationSpeed = 0.25f;
                                /*moveVec.x += -delta.x * distRatio * collisionRelaxationSpeed;
                                moveVec.y += -delta.y * distRatio * collisionRelaxationSpeed;
                                moveVec.z += -delta.z * distRatio * collisionRelaxationSpeed;*/

                                // Align relative orientation
                                float4 targetRelativeOrientationDelta = mul(
                                    quaternion(VECTOR_UP, -PI / 6),
                                    quaternion(VECTOR_RIGHT, PI / 6)
                                );
                                constexpr float relativeOrientationRelaxationSpeed = 0.2f;
                                p.rot = slerp(
                                    p.rot,
                                    getTargetRelativeOrientation(p, tp, targetRelativeOrientationDelta),
                                    relativeOrientationRelaxationSpeed
                                );

                                // Align relative position
                                float4 targetRelativePositionRotation = quaternion(VECTOR_FORWARD, -PI / 6);
                                constexpr float relativePositionRelaxationSpeed = 0.2f;
                                float3 relaxedRelativePosition = getRelaxedRelativePosition(
                                    p,
                                    tp,
                                    targetRelativeOrientationDelta,
                                    targetRelativePositionRotation,
                                    interactionDistance
                                );
                                moveVec.x += (relaxedRelativePosition.x - p.pos.x) * relativePositionRelaxationSpeed;
                                moveVec.y += (relaxedRelativePosition.y - p.pos.y) * relativePositionRelaxationSpeed;
                                moveVec.z += (relaxedRelativePosition.z - p.pos.z) * relativePositionRelaxationSpeed;

                                p.debugVector.x = (relaxedRelativePosition.x - p.pos.x) * (1.0 - relativePositionRelaxationSpeed);
                                p.debugVector.y = (relaxedRelativePosition.y - p.pos.y) * (1.0 - relativePositionRelaxationSpeed);
                                p.debugVector.z = (relaxedRelativePosition.z - p.pos.z) * (1.0 - relativePositionRelaxationSpeed);
                            } else if (interaction.type == 1) {
                                float deltaInteractionDist = -(interactionDistance - dist);
                                constexpr float distanceRelaxationSpeed = 0.2f;
                                //moveVec += normalizedDelta * (deltaInteractionDist * distanceRelaxationSpeed);

                                constexpr float relativeOrientationRelaxationSpeed = 0.2f;
                                // FIXME: the order should be determined not by ids, but by interaction parameters
                                p.rot = slerp(
                                    p.rot,
                                    mul(quaternionFromTo(up, (p.id < tp.id ? -1 : 1) * normalizedDelta), p.rot),
                                    relativeOrientationRelaxationSpeed
                                );

                                float targetRelativePositionAngle = PI;
                                float targetRelativePositionAngleFlex = PI / 8;
                                constexpr float relativePositionRelaxationSpeed = 0.35f;
                                // FIXME: the order should be determined not by ids, but by interaction parameters
                                float3 relPosVec = (p.id < tp.id ? 1 : -1) * (normalizedDelta);
                                float3 relativePositionRelaxationAxis = normalize(cross(tup, relPosVec));
                                float currentRelativePositionAngle = angle(tup, relPosVec);

                                float3 relaxedRelativePosition = (p.id < tp.id ? -1 : 1) * transform_vector(
                                    tup,
                                    quaternion(
                                        relativePositionRelaxationAxis,
                                        min(
                                            max(
                                                currentRelativePositionAngle,
                                                targetRelativePositionAngle - targetRelativePositionAngleFlex
                                            ),
                                            targetRelativePositionAngle + targetRelativePositionAngleFlex
                                        )
                                    )
                                );
                                relaxedRelativePosition *= interactionDistance;
                                relaxedRelativePosition += tp.pos;
                                // FIXME: the order should be determined not by ids, but by interaction parameters
                                //if(k < 1 && p.nActiveInteractions > 1)
                                moveVec += (relaxedRelativePosition - p.pos) * relativePositionRelaxationSpeed;
                                p.debugVector.x = (relaxedRelativePosition.x - p.pos.x) * (1.0 - relativePositionRelaxationSpeed);
                                p.debugVector.y = (relaxedRelativePosition.y - p.pos.y) * (1.0 - relativePositionRelaxationSpeed);
                                p.debugVector.z = (relaxedRelativePosition.z - p.pos.z) * (1.0 - relativePositionRelaxationSpeed);
                            }
                        }
                    }

                    //// Do not process collisions for the interacting particles,
                    //// the interaction alignment code takes care of this
                    //if (!interactionPartners) {
                    //    float collisionDist = p.radius + tp.radius;
                    //    if (dist < collisionDist) {
                    //        float deltaCollisionDist = -max(collisionDist - dist, 0.0f);
                    //        constexpr float collisionRelaxationSpeed = 0.25f;
                    //        moveVec += normalizedDelta * (deltaCollisionDist * collisionRelaxationSpeed);
                    //    }
                    //}
                }
            }
        }
    }

    // Move the particle
    p.pos = clamp(
        add(
            p.pos,
            moveVec
        ),
        0.0f, d_Config.simSize
    );

    //p.rot = normalize(p.rot);

    //p.rot = mul(p.rot, rotQuat);

    nextParticles[idx] = p;
}

template <typename OriginalType, typename ReducedType>
__global__ void
reduceParticles(
    const OriginalType* particles,
    ReducedType* reducedParticles,
    int stepStart_nActiveParticles
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= stepStart_nActiveParticles)
        return;

    OriginalType p = particles[idx];

    ReducedType rp(p);

    reducedParticles[idx] = rp;
}
