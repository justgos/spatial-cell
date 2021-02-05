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
                                    HYDROPHYLIC,
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
    p.posNoise = mul(
        transform_vector(VECTOR_UP, random_rotation(&rngState[idx])),
        fabs(curand_normal(&rngState[idx])) * movementNoiseScale / max(p.radius / 2.5, 1.0f)
    );
    /*p.velocity = add(
        mul(p.velocity, d_Config.velocityDecay),
        mul(
            transform_vector(VECTOR_UP, random_rotation(&rngState[idx])),
            fabs(curand_normal(&rngState[idx])) * movementNoiseScale / max(p.radius / 2.5, 1.0f)
        )
    );*/
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
    p.angularNoise = slerp(
        p.angularNoise,
        random_rotation(&rngState[idx]),
        d_Config.rotationNoiseScale / (p.radius / 2.5)
    );
    /*p.angularVelocity = slerp(
        slerp(p.angularVelocity, QUATERNION_IDENTITY, d_Config.angularVelocityDecay),
        random_rotation(&rngState[idx]),
        d_Config.rotationNoiseScale / (p.radius / 2.5)
    );*/
    
    particles[idx] = p;
}

template <typename T>
__global__ void
applyNoise(
    const int step,
    curandState* rngState,
    T* particles,
    const int stepStart_nActiveParticles
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= stepStart_nActiveParticles)
        return;

    T p = particles[idx];

    if (!(p.flags & PARTICLE_FLAG_ACTIVE)) {
        return;
    }

    p.velocity += p.posNoise;
    p.posNoise = VECTOR_ZERO;

    p.angularVelocity = mul(p.angularNoise, p.angularVelocity);
    p.angularNoise = QUATERNION_IDENTITY;
    
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
    p.velocity *= d_Config.velocityDecay;

    // Rotate the particle
    p.rot = mul(
        slerp(QUATERNION_IDENTITY, p.angularVelocity, stepFraction),
        p.rot
    );
    p.angularVelocity = slerp(p.angularVelocity, QUATERNION_IDENTITY, d_Config.angularVelocityDecay);

    particles[idx] = p;
}


/*
* Noise applied to interaction partners should be aligned
*/
__global__ void
coordinateNoise(
    const int step,
    curandState* rngState,
    const Particle* curParticles,
    Particle* nextParticles,
    int stepStart_nActiveParticles,
    unsigned int* gridRanges,
    float coordinationFraction
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

                        // Non-rigid interactions (e.g. RNA chain) should be less coordinated,
                        // otherwise their behavior becomes too rigid
                        if (interaction.type < 10 && coordinationFraction > 0.25f)
                            continue;

                        if (interaction.partnerId == tp.id) {
                            // Noise should act on the interacting partners as if they were one
                            float pPosNoiseMag = length(p.posNoise);
                            float tpPosNoiseMag = length(tp.posNoise);
                            p.posNoise = lerp(p.posNoise, tp.posNoise, 0.9f * tp.radius / (p.radius + tp.radius) / (p.nActiveInteractions + tp.nActiveInteractions));
                            // Restore some of the impulse
                            p.posNoise *= 1.0f + ((pPosNoiseMag * p.radius + tpPosNoiseMag * tp.radius) / (p.radius + tp.radius) / length(p.posNoise) - 1.0f) * 0.5f;

                            //p.velocity = p.velocity - -normalizedDelta * min(dot(p.velocity, -normalizedDelta), 0.0) * 0.5;

                            p.angularNoise = slerp(
                                p.angularNoise,
                                tp.angularNoise,
                                0.9f * tp.radius / (p.radius + tp.radius) / p.nActiveInteractions  //(p.nActiveInteractions + tp.nActiveInteractions)
                            );

                            // Angular noise is too messy for particle complexes - it'll be taken care of
                            // when the particles will attempt to realign after applying the position noise
                            /*p.angularVelocity = slerp(
                                p.angularVelocity,
                                QUATERNION_IDENTITY,
                                0.4f / (p.nActiveInteractions + tp.nActiveInteractions)
                            );*/
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

    float3 moveVec = VECTOR_ZERO;
    float3 dVelocity = VECTOR_ZERO;
    float4 dAngularVelocity = QUATERNION_IDENTITY;
    float3 phobicAttractionPos = VECTOR_ZERO;
    float phobicAttractionPowerSum = 0.0f;
    //float4 rotQuat = QUATERNION_IDENTITY;
    float3 targetPos = p.pos;
    float countedInteractions = 0.0f;
    float countedInteractionWeight = 0.0f;

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

                    if ((p.hydrophobic > 0 || tp.hydrophobic > 0) && dist <= p.radius + tp.radius + 0.8 * (p.radius + tp.radius) * 0.5f) {
                        if (p.hydrophobic == POLAR && tp.hydrophobic == POLAR) {
                            float3 pPhobicDir = -up;
                            float3 tpPhobicDir = -tup;
                            float3 pPhobicDirVert = pPhobicDir * p.radius + p.pos;
                            float3 tpPhobicDirVert = tpPhobicDir * tp.radius + tp.pos;
                            float3 pPhilicDir = up;
                            float3 tpPhilicDir = tup;
                            float3 pPhilicDirVert = pPhilicDir * p.radius + p.pos;
                            float3 tpPhilicDirVert = tpPhilicDir * tp.radius + tp.pos;
                            // TODO: reduce for longer-range interactions
                            //float attractionPower = (dot(pPhobicDir, normalizedDelta) + 1.0f) * 0.5f * (dot(tpPhobicDir, -normalizedDelta) + 1.0f) * 0.5f;
                            float powerCap = 10.0f;
                            float attractionPower = min(max(length(tpPhobicDirVert - pPhobicDirVert) / (p.radius + tp.radius), 1e-6), 2.0f) / 2.0f;
                            float repulsionPower = min(1.0f / max(length(tpPhilicDirVert - pPhobicDirVert) / (p.radius + tp.radius), 1e-6), powerCap) / powerCap;
                            constexpr float distanceRelaxationSpeed = 0.05f;
                            dVelocity += (tpPhobicDirVert + tp.velocity - (pPhobicDirVert + p.velocity)) * distanceRelaxationSpeed * attractionPower;
                            //dVelocity += (tp.pos - normalizedDelta * interactionDistance + tp.velocity - (p.pos + p.velocity)) * distanceRelaxationSpeed;
                            countedInteractionWeight += 1.0f;

                            constexpr float orientationRelaxationSpeed = 0.15f;
                            /*dAngularVelocity = mul(
                                slerp(
                                    QUATERNION_IDENTITY,
                                    quaternionFromTo(pPhobicDir, normalizedDelta),
                                    orientationRelaxationSpeed * attractionPower
                                ),
                                dAngularVelocity
                            );*/

                            float phobicAttractionPower = pow(min(1.0f / max(length(tpPhobicDirVert - pPhobicDirVert) / (p.radius + tp.radius), 1e-6f), powerCap) / powerCap, 0.2f);
                            phobicAttractionPowerSum += phobicAttractionPower;
                            phobicAttractionPos += tpPhobicDirVert * phobicAttractionPower;

                            float targetRelativePositionAngle = PI / 2;
                            float maxRelativeOrientationDeviation = PI / 2.0f - PI / 12.0f;
                            /*float3 orientationRelaxationAxis = safeCross(pPhobicDir, normalizedDelta);
                            float currentRelativeOrientationAngle = angle(pPhobicDir, normalizedDelta);
                            float3 relaxedRelativeOrientation = transform_vector(
                                pPhobicDir,
                                quaternion(
                                    orientationRelaxationAxis,
                                    min(
                                        max(
                                            currentRelativeOrientationAngle,
                                            currentRelativeOrientationAngle
                                        ),
                                        currentRelativeOrientationAngle
                                    )
                                )
                            );*/

                            /*dAngularVelocity = slerp(
                                dAngularVelocity,
                                quaternionFromTo(pPhobicDir, normalize(tpPhobicDirVert - p.pos)),
                                orientationRelaxationSpeed * attractionPower
                            );
                            dAngularVelocity = slerp(
                                dAngularVelocity,
                                inverse(quaternionFromTo(pPhobicDir, normalize(tpPhilicDirVert - p.pos))),
                                orientationRelaxationSpeed * repulsionPower
                            );
                            dAngularVelocity = slerp(
                                dAngularVelocity,
                                inverse(quaternionFromTo(pPhilicDir, normalize(tpPhobicDirVert - p.pos))),
                                orientationRelaxationSpeed * min(1.0f / max(length(tpPhobicDirVert - pPhilicDirVert) / (p.radius + tp.radius), 1e-6), powerCap) / powerCap
                            );*/

                            /*p.rot = slerp(
                                p.rot,
                                mul(relaxedRelativeOrientation, p.rot),
                                orientationRelaxationSpeed
                            );*/
                            /*dAngularVelocity = slerp(
                                dAngularVelocity,
                                mul(quaternionFromTo(tpPhobicVec, relaxedRelativeOrientation), p.rot),
                                orientationRelaxationSpeed * attractionPower
                            );*/

                            ////float coalignment = (dot(pPhobicDir, tpPhobicDir) + 1.0f) * 0.5f;
                            //float coalignment = max(dot(pPhobicDir, tpPhobicDir), 0.0f);
                            ////coalignment *= coalignment;
                            //dAngularVelocity = slerp(
                            //    dAngularVelocity,
                            //    quaternionFromTo(up, tup),
                            //    orientationRelaxationSpeed * coalignment
                            //);

                            //dAngularVelocity = slerp(
                            //    dAngularVelocity,
                            //    quaternionFromTo(
                            //        up,
                            //        transform_vector(
                            //            normalizedDelta,
                            //            quaternion(
                            //                safeCross(normalizedDelta, up),
                            //                targetRelativePositionAngle
                            //            )
                            //        )
                            //    ),
                            //    orientationRelaxationSpeed * coalignment
                            //);
                        }
                    }

                    bool interactionPartners = false;

                    for (int k = 0; k < p.nActiveInteractions; k++) {
                        ParticleInteraction interaction = p.interactions[k];
                        if (interaction.partnerId == tp.id) {
                            interactionPartners = true;

                            if (interaction.type >= 10) {
                                ParticleInteractionInfo pii = flatComplexificationInfo[interaction.type];

                                // Reduce the noise coming from the direction of the partner (the opposite-facing component), 
                                // as there're less noise water molecules there
                                //p.velocity = p.velocity - -normalizedDelta * min(dot(p.velocity, -normalizedDelta), 0.0) * 0.2;

                                // Align relative orientation
                                constexpr float relativeOrientationRelaxationSpeed = 0.1f;
                                dAngularVelocity = slerp(
                                    dAngularVelocity,
                                    mul(inverse(mul(p.angularVelocity, p.rot)), mul(tp.angularVelocity, getTargetRelativeOrientation(p, tp, pii.relativeOrientation))),
                                    relativeOrientationRelaxationSpeed
                                );
                                /*p.rot = slerp(
                                    p.rot,
                                    getTargetRelativeOrientation(p, tp, pii.relativeOrientation),
                                    relativeOrientationRelaxationSpeed
                                );*/

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
                                //moveVec += relativePositionDelta * 0.99f / (p.nActiveInteractions + tp.nActiveInteractions);  // *(p.radius / (p.radius + tp.radius));
                                float interactionWeight = tp.radius;
                                countedInteractions += 1.0f;
                                countedInteractionWeight += interactionWeight;
                                //targetPos += (relaxedRelativePosition - targetPos) * (interactionWeight / countedInteractionWeight);
                                //p.velocity += (relaxedRelativePosition - targetPos) * 0.1f * (interactionWeight / countedInteractionWeight);
                                dVelocity += (relaxedRelativePosition + tp.velocity - (p.pos + p.velocity)) * 0.45f / p.nActiveInteractions * tp.radius / max(p.radius, tp.radius);  // / (p.radius + tp.radius);
                                //targetPos += (relaxedRelativePosition - targetPos) / countedInteractions;
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
                                moveVec += (relaxedRelativePosition - p.pos) * relativePositionRelaxationSpeed;

                                p.debugVector.x += (relaxedRelativePosition.x - p.pos.x) * (1.0 - relativePositionRelaxationSpeed);
                                p.debugVector.y += (relaxedRelativePosition.y - p.pos.y) * (1.0 - relativePositionRelaxationSpeed);
                                p.debugVector.z += (relaxedRelativePosition.z - p.pos.z) * (1.0 - relativePositionRelaxationSpeed);
                            } else if (interaction.type == 1) {
                                // Flexible chain interaction
                                float deltaInteractionDist = -(interactionDistance - dist);
                                constexpr float distanceRelaxationSpeed = 0.2f;
                                //moveVec += normalizedDelta * (deltaInteractionDist * distanceRelaxationSpeed);

                                constexpr float relativeOrientationRelaxationSpeed = 0.2f;
                                dAngularVelocity = slerp(
                                    dAngularVelocity,
                                    quaternionFromTo(transform_vector(up, p.angularVelocity), (interaction.group == INTERACTION_GROUP_FORWARD ? -1 : 1) * normalizedDelta),
                                    relativeOrientationRelaxationSpeed
                                );
                                /*p.rot = slerp(
                                    p.rot,
                                    mul(quaternionFromTo(up, (interaction.group == INTERACTION_GROUP_FORWARD ? -1 : 1) * normalizedDelta), p.rot),
                                    relativeOrientationRelaxationSpeed
                                );*/

                                float targetRelativePositionAngle = PI;
                                float targetRelativePositionAngleFlex = PI / 8;
                                constexpr float relativePositionRelaxationSpeed = 0.35f;
                                // FIXME: the order should be determined not by ids, but by interaction parameters
                                float3 relPosVec = (p.id < tp.id ? 1 : -1) * (normalizedDelta);
                                float3 relativePositionRelaxationAxis = safeCross(tup, relPosVec);
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

                                //moveVec += (relaxedRelativePosition - p.pos) * relativePositionRelaxationSpeed;
                                //float3 pJ = relaxedRelativePosition - p.pos;
                                dVelocity += (relaxedRelativePosition + tp.velocity - (p.pos + p.velocity)) * 0.45f / p.nActiveInteractions * tp.radius / max(p.radius, tp.radius);  // / (p.radius + tp.radius);
                                /*float interactionWeight = tp.radius;
                                countedInteractions += 1.0f;
                                countedInteractionWeight += interactionWeight;
                                targetPos += (relaxedRelativePosition - targetPos) * (interactionWeight / countedInteractionWeight);*/

                                p.debugVector.x += (relaxedRelativePosition.x - p.pos.x) * (1.0 - relativePositionRelaxationSpeed);
                                p.debugVector.y += (relaxedRelativePosition.y - p.pos.y) * (1.0 - relativePositionRelaxationSpeed);
                                p.debugVector.z += (relaxedRelativePosition.z - p.pos.z) * (1.0 - relativePositionRelaxationSpeed);
                            }
                        }
                    }

                    // Do not process collisions for the interacting particles,
                    // the interaction alignment code takes care of this
                    if (!interactionPartners) {
                        float collisionDist = p.radius + tp.radius;
                        if (dist < collisionDist) {
                            //float deltaCollisionDist = -max(collisionDist - dist, 0.0f);
                            //constexpr float collisionRelaxationSpeed = 0.25f;
                            //moveVec += normalizedDelta * (deltaCollisionDist * collisionRelaxationSpeed) * p.radius / (p.radius + tp.radius);
                            dVelocity += (tp.pos - normalizedDelta * collisionDist + tp.velocity - (p.pos + p.velocity)) * 0.5f * tp.radius / max(p.radius, tp.radius);  // / (p.radius + tp.radius);;
                            countedInteractionWeight += 1.0f;
                        }
                    }
                }
            }
        }
    }

    //// Move the particle
    //p.pos = clamp(
    //    p.pos + (targetPos - p.pos) * 0.49f + moveVec,
    //    0.0f, d_Config.simSize
    //);
    //p.velocity = p.velocity + ((targetPos - p.pos) * 0.49f + moveVec) * 0.5f,

    p.velocity += dVelocity / max(countedInteractionWeight, 1.0f);
    p.angularVelocity = mul(
        slerp(QUATERNION_IDENTITY, dAngularVelocity, 1.0f / max(countedInteractionWeight, 1.0f)),
        p.angularVelocity
    );

    if (phobicAttractionPowerSum > 0.0f) {
        float3 phobicAttractionDir = normalize(phobicAttractionPos / phobicAttractionPowerSum - p.pos);
        /*p.debugVector.x = phobicAttractionDir.x * -0.5f;
        p.debugVector.y = phobicAttractionDir.y * -0.5f;
        p.debugVector.z = phobicAttractionDir.z * -0.5f;*/
        p.angularVelocity = slerp(
            p.angularVelocity,
            quaternionFromTo(-up, phobicAttractionDir),
            0.2f
        );
    }

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
