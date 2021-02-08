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


std::unordered_map<int, ParticleInteractionInfo>*
loadComplexificationInfo() {
    auto complexificationInfo = new std::unordered_map<int, ParticleInteractionInfo>();

    std::ifstream complexificationFile("../universe-config/gen/complexification.json");
    Json::Value complexificationJson;
    complexificationFile >> complexificationJson;

    for (int i = 0; i < complexificationJson.size(); i++) {
        auto c = complexificationJson[i];
        int id = c["id"].asInt();

        auto relativeOrientation = c["relativeOrientation"];
        auto relativePosition = c["relativePosition"];
        complexificationInfo->insert(std::pair<int, ParticleInteractionInfo>(id, ParticleInteractionInfo(
            id,
            c.get("group", -1).asInt(),
            c["firstPartnerType"].asInt(),
            c.get("firstPartnerState", -1).asInt(),
            c["secondPartnerType"].asInt(),
            c.get("secondPartnerState", -1).asInt(),
            c.get("propagateState", false).asBool(),
            c.get("setState", true).asBool(),
            c.get("waitForState", true).asBool(),
            c.get("waitForAlignment", true).asBool(),
            c.get("onlyViaTransition", false).asBool(),
            c.get("transitionTo", -1).asInt(),
            c.get("polymerize", -1).asInt(),
            c.get("breakAfterTransition", false).asBool(),
            make_float4(relativeOrientation[0].asFloat(), relativeOrientation[1].asFloat(), relativeOrientation[2].asFloat(), relativeOrientation[3].asFloat()),
            make_float3(relativePosition[0].asFloat(), relativePosition[1].asFloat(), relativePosition[2].asFloat())
        )));
    }

    return complexificationInfo;
}

/*
* Transform the interaction map into a flat array with index<->id identity
*/
SingleBuffer<ParticleInteractionInfo>*
flattenComplexificationInfo(std::unordered_map<int, ParticleInteractionInfo>* complexificationInfo) {
    int maxIndex = 0;
    for (auto it = complexificationInfo->begin(); it != complexificationInfo->end(); it++) {
        maxIndex = max(it->first, maxIndex);
    }

    auto flattenedInfo = new SingleBuffer<ParticleInteractionInfo>(maxIndex + 1);

    for (auto it = complexificationInfo->begin(); it != complexificationInfo->end(); it++) {
        flattenedInfo->h_Current[it->first] = it->second;
    }

    return flattenedInfo;
}

/*
* Transform the interaction map into a flat index ordered by the participant id
* (each interaction is present two times - for each participant)
*/
std::pair<
    SingleBuffer<int>*,
    SingleBuffer<ParticleInteractionInfo>*
>
partnerMapComplexificationInfo(std::unordered_map<int, ParticleInteractionInfo>* complexificationInfo) {
    int maxIndex = 0;
    for (auto it = complexificationInfo->begin(); it != complexificationInfo->end(); it++) {
        maxIndex = max(it->first, maxIndex);
    }

    // Convert interaction_id->interaction_info map into partner_id->interaction_info map
    auto partnerMappedInfoMap = new std::map<int, std::vector<ParticleInteractionInfo>*>();
    for (auto it = complexificationInfo->begin(); it != complexificationInfo->end(); it++) {
        ParticleInteractionInfo pii = it->second;
        std::vector<ParticleInteractionInfo> *interactionsForType;

        if (partnerMappedInfoMap->count(pii.firstPartnerType)) {
            interactionsForType = (*partnerMappedInfoMap)[pii.firstPartnerType];
        } else {
            interactionsForType = new std::vector<ParticleInteractionInfo>();
            partnerMappedInfoMap->insert(std::pair<int, std::vector<ParticleInteractionInfo>*>(pii.firstPartnerType, interactionsForType));
        }
        interactionsForType->push_back(pii);

        if (partnerMappedInfoMap->count(pii.secondPartnerType)) {
            interactionsForType = (*partnerMappedInfoMap)[pii.secondPartnerType];
        }
        else {
            interactionsForType = new std::vector<ParticleInteractionInfo>();
            partnerMappedInfoMap->insert(std::pair<int, std::vector<ParticleInteractionInfo>*>(pii.secondPartnerType, interactionsForType));
        }
        interactionsForType->push_back(pii);
    }

    auto partnerIndexMap = new SingleBuffer<int>((maxIndex+1) * 2);
    auto partnerMappedInfo = new SingleBuffer<ParticleInteractionInfo>(complexificationInfo->size() * 2);

    // Flatten the partner-mapped interactions
    int n = 0;
    for (auto it = partnerMappedInfoMap->begin(); it != partnerMappedInfoMap->end(); it++) {
        // Start and end indices of the interaction block for the `it->first` particle type
        partnerIndexMap->h_Current[it->first * 2] = n;
        partnerIndexMap->h_Current[it->first * 2 + 1] = n + it->second->size();

        for (int i = 0; i < it->second->size(); i++) {
            partnerMappedInfo->h_Current[n] = (*it->second)[i];
            n++;
        }
    }

    for (auto it = partnerMappedInfoMap->begin(); it != partnerMappedInfoMap->end(); it++)
        delete it->second;
    delete partnerMappedInfoMap;

    return std::pair<
        SingleBuffer<int>*,
        SingleBuffer<ParticleInteractionInfo>*
    >(
        partnerIndexMap,
        partnerMappedInfo
    );
}


std::unordered_map<int, ComplexInfo>*
loadComplexInfo() {
    auto complexInfo = new std::unordered_map<int, ComplexInfo>();

    std::ifstream complexFile("../universe-config/gen/complexes.json");
    Json::Value complexJson;
    complexFile >> complexJson;

    for (int i = 0; i < complexJson.size(); i++) {
        auto c = complexJson[i];
        int id = c["id"].asInt();

        int nParticipants = c["participants"].size();
        ComplexParticipantInfo* participants = new ComplexParticipantInfo[nParticipants];
        for (int j = 0; j < nParticipants; j++) {
            auto cp = c["participants"][j];

            auto position = cp["position"];
            auto rotation = cp["rotation"];
            participants[j] = ComplexParticipantInfo(
                cp["type"].asInt(),
                make_float3(position[0].asFloat(), position[1].asFloat(), position[2].asFloat()),
                make_float4(rotation[0].asFloat(), rotation[1].asFloat(), rotation[2].asFloat(), rotation[3].asFloat())
            );
        }

        complexInfo->insert(std::pair<int, ComplexInfo>(id, ComplexInfo(
            nParticipants,
            participants
        )));
    }

    return complexInfo;
}




__device__ __inline__ bool
interactionParticipantOrder(Particle p, Particle tp) {
    return p.type < tp.type || (p.type == tp.type && p.id < tp.id);
}

__device__ __inline__ float4
getTargetRelativeOrientation(Particle p, Particle tp, float4 targetRelativeOrientation) {
	float4 relativeOrientationDelta = interactionParticipantOrder(p, tp)
        ? mul(tp.rot, inverse(targetRelativeOrientation))
        : mul(tp.rot, targetRelativeOrientation);
    return relativeOrientationDelta;
}

__device__ __inline__ float3
getRelaxedRelativePosition(
    Particle p, 
    Particle tp, 
    float4 targetRelativeOrientation, 
    float4 targetRelativePositionRotation, 
    float interactionDistance
) {
    float3 relaxedRelativePosition = interactionParticipantOrder(p, tp)
        ? -transform_vector(VECTOR_UP, mul(tp.rot, mul(inverse(targetRelativeOrientation), targetRelativePositionRotation)))
        : transform_vector(VECTOR_UP, mul(tp.rot, targetRelativePositionRotation));
    relaxedRelativePosition *= interactionDistance;
    relaxedRelativePosition += tp.pos;
    return relaxedRelativePosition;
}


__global__ void
complexify(
    const int step,
    curandState* rngState,
    const Particle* curParticles,
    Particle* nextParticles,
    int stepStart_nActiveParticles,
    unsigned int* gridRanges,
    ParticleInteractionInfo* flatComplexificationInfo,
    int* partnerMappedComplexificationIndex,
    ParticleInteractionInfo *partnerMappedComplexificationInfo
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= stepStart_nActiveParticles)
        return;

    Particle p = curParticles[idx];

    if (!(p.flags & PARTICLE_FLAG_ACTIVE)) {
        nextParticles[idx] = p;
        return;
    }

    // Grid cell index of the current particle
    const int cgx = getGridIdx(p.pos.x),
        cgy = getGridIdx(p.pos.y),
        cgz = getGridIdx(p.pos.z);

    // Up direction of the current particle
    float3 up = transform_vector(VECTOR_UP, p.rot);

    int complexificationStartIndex = partnerMappedComplexificationIndex[p.type * 2];
    int complexificationEndIndex = partnerMappedComplexificationIndex[p.type * 2 + 1];

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

                    for (int k = complexificationStartIndex; k < complexificationEndIndex; k++) {
                        // The particle can't have any more interactions
                        if (p.nActiveInteractions >= MAX_ACTIVE_INTERACTIONS)
                            continue;

                        ParticleInteractionInfo pii = partnerMappedComplexificationInfo[k];

                        // Check whether these two particles are suitable partners
                        if (!(
                                (p.type == pii.firstPartnerType && tp.type == pii.secondPartnerType)
                                || (p.type == pii.secondPartnerType && tp.type == pii.firstPartnerType)
                            )
                            || pii.onlyViaTransition
                        )
                            continue;

                        // Check whether this interaction group is already active
                        // TODO: prevent two different particles forming an interaction with a third one during a single step
                        bool interactionGroupAlreadyActive = false;
                        if (pii.group >= 0) {
                            for (int k = 0; k < p.nActiveInteractions; k++) {
                                ParticleInteraction interaction = p.interactions[k];
                                if (interaction.group == pii.group) {
                                    interactionGroupAlreadyActive = true;
                                    break;
                                }
                            }
                            for (int k = 0; k < tp.nActiveInteractions; k++) {
                                ParticleInteraction interaction = tp.interactions[k];
                                if (interaction.group == pii.group) {
                                    interactionGroupAlreadyActive = true;
                                    break;
                                }
                            }
                        }
                        if (interactionGroupAlreadyActive)
                            continue;

                        float orientationAlignment = interactionParticipantOrder(p, tp)
                            ? dot(transform_vector(up, pii.relativeOrientation), tup)
                            : dot(transform_vector(tup, pii.relativeOrientation), up);
                        float relativePositionAlignment = interactionParticipantOrder(p, tp)
                            ? dot(transform_vector(pii.relativePosition, p.rot), delta) / normsq(delta)
                            : dot(transform_vector(pii.relativePosition, tp.rot), -delta) / normsq(delta);

                        // If the particles are well-aligned - create the interaction
                        if (
                            orientationAlignment > 0.8
                            && fabs(relativePositionAlignment - 1.0) < 0.2
                            && (interactionParticipantOrder(p, tp)
                                ? ((pii.firstPartnerState < 0 || pii.firstPartnerState == p.state) && (pii.secondPartnerState < 0 || pii.secondPartnerState == tp.state))
                                : ((pii.secondPartnerState < 0 || pii.secondPartnerState == p.state) && (pii.firstPartnerState < 0 || pii.firstPartnerState == tp.state)))
                        ) {
                            p.interactions[p.nActiveInteractions].type = pii.id;
                            p.interactions[p.nActiveInteractions].group = pii.group;
                            p.interactions[p.nActiveInteractions].partnerId = tp.id;
                            p.nActiveInteractions++;
                        }
                    }
                }
            }
        }
    }

    nextParticles[idx] = p;
}


__global__ void
transitionInteractions(
    const int step,
    curandState* rngState,
    const Particle* curParticles,
    Particle* nextParticles,
    int stepStart_nActiveParticles,
    int* nActiveParticles,
    int* lastActiveParticle,
    int* nextParticleId,
    unsigned int* gridRanges,
    MinimalParticleTypeInfo* flatParticleTypeInfo,
    ParticleInteractionInfo* flatComplexificationInfo,
    int* partnerMappedComplexificationIndex,
    ParticleInteractionInfo* partnerMappedComplexificationInfo
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= stepStart_nActiveParticles)
        return;

    Particle p = curParticles[idx];

    if (!(p.flags & PARTICLE_FLAG_ACTIVE)) {
        nextParticles[idx] = p;
        return;
    }

    // Grid cell index of the current particle
    const int cgx = getGridIdx(p.pos.x),
        cgy = getGridIdx(p.pos.y),
        cgz = getGridIdx(p.pos.z);

    // Up direction of the current particle
    float3 up = transform_vector(VECTOR_UP, p.rot);

    int complexificationStartIndex = partnerMappedComplexificationIndex[p.type * 2];
    int complexificationEndIndex = partnerMappedComplexificationIndex[p.type * 2 + 1];

    int sourceTransitionIndex = -1;
    ParticleInteractionInfo targetTransition;

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

                    for (int k = 0; k < p.nActiveInteractions; k++) {
                        ParticleInteraction interaction = p.interactions[k];

                        if (interaction.type < 10)
                            continue;

                        if (interaction.partnerId == tp.id) {
                            ParticleInteractionInfo pii = flatComplexificationInfo[interaction.type];

                            // Check if there are any possible transitions here
                            if (!(
                                    pii.transitionTo >= 0
                                    || pii.breakAfterTransition
                                )
                            )
                                continue;

                            float orientationAlignment = interactionParticipantOrder(p, tp)
                                ? dot(transform_vector(up, pii.relativeOrientation), tup)
                                : dot(transform_vector(tup, pii.relativeOrientation), up);
                            float relativePositionAlignment = interactionParticipantOrder(p, tp)
                                ? dot(transform_vector(pii.relativePosition, p.rot), delta) / normsq(delta)
                                : dot(transform_vector(pii.relativePosition, tp.rot), -delta) / normsq(delta);

                            // If the particles are well-aligned - create the interaction
                            if (
                                (pii.breakAfterTransition || !pii.waitForAlignment
                                    || (orientationAlignment > 0.8
                                        && fabs(relativePositionAlignment - 1.0) < 0.2)
                                )
                                && (!pii.waitForState || 
                                    (interactionParticipantOrder(p, tp)
                                        ? ((pii.firstPartnerState < 0 || pii.firstPartnerState == p.state) && (pii.secondPartnerState < 0 || pii.secondPartnerState == tp.state))
                                        : ((pii.secondPartnerState < 0 || pii.secondPartnerState == p.state) && (pii.firstPartnerState < 0 || pii.firstPartnerState == tp.state))
                                    )
                                   )
                            ) {
                                if (pii.transitionTo >= 0) {
                                    ParticleInteractionInfo tpii = flatComplexificationInfo[pii.transitionTo];

                                    // Check whether this interaction group is already active
                                    // TODO: prevent two different particles forming an interaction with a third one during a single step
                                    bool interactionGroupAlreadyActive = false;
                                    if (tpii.group >= 0 && tpii.group != pii.group) {
                                        for (int l = 0; l < p.nActiveInteractions; l++) {
                                            ParticleInteraction interaction = p.interactions[l];
                                            if (interaction.group == tpii.group) {
                                                interactionGroupAlreadyActive = true;
                                                break;
                                            }
                                        }
                                        for (int l = 0; l < tp.nActiveInteractions; l++) {
                                            ParticleInteraction interaction = tp.interactions[l];
                                            if (interaction.group == tpii.group) {
                                                interactionGroupAlreadyActive = true;
                                                break;
                                            }
                                        }
                                    }
                                    if (interactionGroupAlreadyActive)
                                        continue;

                                    if (!tpii.setState ||
                                        !((p.type == tpii.firstPartnerType && tpii.firstPartnerState >= 0)
                                            || (p.type == tpii.secondPartnerType && tpii.secondPartnerState >= 0)
                                        )
                                    ) {
                                        // If this transition won't affect this particle's state - play it immediately
                                        p.interactions[k].type = tpii.id;
                                        p.interactions[k].group = tpii.group;
                                    } else {
                                        sourceTransitionIndex = k;
                                        targetTransition = tpii;
                                    }
                                    //p.interactions[k].type = tpii.id;
                                    //p.interactions[k].group = tpii.group;
                                    ////p.interactions[k].partnerId = tp.id;

                                    //if (tpii.setState) {
                                    //    // Update the particle's state to match the target interaction
                                    //    if (p.type == tpii.firstPartnerType && tpii.firstPartnerState >= 0)
                                    //        p.state = tpii.firstPartnerState;
                                    //    if (p.type == tpii.secondPartnerType && tpii.secondPartnerState >= 0)
                                    //        p.state = tpii.secondPartnerState;
                                    //}
                                } else if (pii.breakAfterTransition) {
                                    // Break this interaction and step back the interaction loop
                                    for (int l = k; l < p.nActiveInteractions - 1; l++) {
                                        p.interactions[l] = p.interactions[l+1];
                                    }
                                    p.nActiveInteractions--;
                                    if (sourceTransitionIndex > k)
                                        sourceTransitionIndex--;
                                    k--;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (sourceTransitionIndex >= 0) {
        p.interactions[sourceTransitionIndex].type = targetTransition.id;
        p.interactions[sourceTransitionIndex].group = targetTransition.group;
        //p.interactions[sourceTransitionIndex].partnerId = tp.id;

        if (targetTransition.setState) {
            // Update the particle's state to match the target interaction
            if (p.type == targetTransition.firstPartnerType && targetTransition.firstPartnerState >= 0) {
                p.state = targetTransition.firstPartnerState;
                p.lastStateChangeStep = step;
            }
            if (p.type == targetTransition.secondPartnerType && targetTransition.secondPartnerState >= 0) {
                p.state = targetTransition.secondPartnerState;
                p.lastStateChangeStep = step;
            }
        }

        if (targetTransition.polymerize >= 0) {
            ParticleInteractionInfo polymerization = flatComplexificationInfo[targetTransition.polymerize];
            
            // The current particle should also be a participant of the target polymerization interaction
            if (
                polymerization.firstPartnerType == p.type
                || polymerization.secondPartnerType == p.type
            ) {
                int newParticleType = polymerization.firstPartnerType == p.type
                    ? polymerization.secondPartnerType
                    : polymerization.firstPartnerType;
                int newParticleState = polymerization.firstPartnerType == p.type
                    ? polymerization.secondPartnerState
                    : polymerization.firstPartnerState;
                MinimalParticleTypeInfo newParticleInfo = flatParticleTypeInfo[newParticleType];
                int newIdx = atomicAdd(lastActiveParticle, 1);
                if (newIdx < d_Config.numParticles) {
                    int newId = atomicAdd(nextParticleId, 1);
                    Particle np = Particle(
                        newId,
                        newParticleType,
                        0,
                        newParticleInfo.radius,
                        newParticleInfo.hydrophobic,
                        p.pos + transform_vector(
                            p.type < newParticleType
                            ? polymerization.relativePosition
                            : -polymerization.relativePosition,
                            p.rot
                        ),
                        mul(
                            p.rot,
                            p.type < newParticleType
                            ? polymerization.relativeOrientation
                            : inverse(polymerization.relativeOrientation)
                        )
                    );
                    if (newParticleState >= 0) {
                        np.state = newParticleState;
                        np.lastStateChangeStep = step;
                    }
                    np.interactions[np.nActiveInteractions].type = targetTransition.polymerize;
                    np.interactions[np.nActiveInteractions].group = polymerization.group;
                    np.interactions[np.nActiveInteractions].partnerId = p.id;
                    np.nActiveInteractions++;

                    int activePolymerizationIndex = -1;
                    // Does the current particle already have a polymerication interaction active?
                    for (int k = 0; k < p.nActiveInteractions; k++) {
                        ParticleInteraction interaction = p.interactions[k];
                        if (interaction.group == polymerization.group) {
                            activePolymerizationIndex = k;
                            break;
                        }
                    }
                    if (activePolymerizationIndex >= 0) {
                        // Transfer the existing interaction to a new partner
                        p.interactions[activePolymerizationIndex].partnerId = np.id;
                    } else {
                        p.interactions[p.nActiveInteractions].type = targetTransition.polymerize;
                        p.interactions[p.nActiveInteractions].group = polymerization.group;
                        p.interactions[p.nActiveInteractions].partnerId = np.id;
                        p.nActiveInteractions++;
                    }

                    nextParticles[newIdx] = np;
                    atomicAdd(nActiveParticles, 1);
                }
            }
        }
    }

    nextParticles[idx] = p;
}


__global__ void
propagateState(
    const int step,
    curandState* rngState,
    const Particle* curParticles,
    Particle* nextParticles,
    int stepStart_nActiveParticles,
    unsigned int* gridRanges,
    ParticleInteractionInfo* flatComplexificationInfo,
    int* partnerMappedComplexificationIndex,
    ParticleInteractionInfo* partnerMappedComplexificationInfo
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= stepStart_nActiveParticles)
        return;

    Particle p = curParticles[idx];

    if (!(p.flags & PARTICLE_FLAG_ACTIVE)) {
        nextParticles[idx] = p;
        return;
    }

    // Grid cell index of the current particle
    const int cgx = getGridIdx(p.pos.x),
        cgy = getGridIdx(p.pos.y),
        cgz = getGridIdx(p.pos.z);

    // Up direction of the current particle
    float3 up = transform_vector(VECTOR_UP, p.rot);

    int complexificationStartIndex = partnerMappedComplexificationIndex[p.type * 2];
    int complexificationEndIndex = partnerMappedComplexificationIndex[p.type * 2 + 1];

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

                    float3 delta = tp.pos - p.pos;
                    // Skip particles beyond the maximum interaction distance
                    if (fabs(delta.x) > d_Config.maxInteractionDistance
                        || fabs(delta.y) > d_Config.maxInteractionDistance
                        || fabs(delta.z) > d_Config.maxInteractionDistance)
                        continue;

                    for (int k = 0; k < p.nActiveInteractions; k++) {
                        ParticleInteraction interaction = p.interactions[k];

                        if (interaction.type < 10)
                            continue;

                        if (interaction.partnerId == tp.id) {
                            ParticleInteractionInfo pii = flatComplexificationInfo[interaction.type];

                            if (!pii.propagateState)
                                continue;

                            if (tp.lastStateChangeStep > p.lastStateChangeStep) {
                                p.lastStateChangeStep = tp.lastStateChangeStep;
                                p.state = tp.state;
                            }
                        }
                    }
                }
            }
        }
    }

    nextParticles[idx] = p;
}
