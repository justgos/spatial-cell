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
            c["firstPartnerType"].asInt(),
            c["secondPartnerType"].asInt(),
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
partnerMappedComplexificationInfo(std::unordered_map<int, ParticleInteractionInfo>* complexificationInfo) {
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
                        ParticleInteractionInfo pii = partnerMappedComplexificationInfo[k];

                        // Check whether these two particles are suitable partners
                        if (!(
                            (p.type == pii.firstPartnerType && tp.type == pii.secondPartnerType)
                            || (p.type == pii.secondPartnerType && tp.type == pii.firstPartnerType)
                        ))
                            continue;

                        // Check whether this interaction is already active
                        bool interactionAlreadyActive = false;
                        for (int k = 0; k < p.nActiveInteractions; k++) {
                            ParticleInteraction interaction = p.interactions[k];
                            if (interaction.type == pii.id) {
                                interactionAlreadyActive = true;
                                break;
                            }
                        }
                        if (interactionAlreadyActive)
                            continue;

                        float orientationAlignment = interactionParticipantOrder(p, tp)
                            ? dot(transform_vector(up, pii.relativeOrientation), tup)
                            : dot(transform_vector(tup, pii.relativeOrientation), up);
                        float relativePositionAlignment = interactionParticipantOrder(p, tp)
                            ? dot(transform_vector(pii.relativePosition, mul(inverse(pii.relativeOrientation), tp.rot)), delta) / normsq(delta)
                            : dot(transform_vector(pii.relativePosition, tp.rot), -delta) / normsq(delta);

                        // If the particles are well-aligned - create an interaction
                        if (
                            orientationAlignment > 0.9
                            && fabs(relativePositionAlignment - 1.0) < 0.1
                        ) {
                            p.interactions[p.nActiveInteractions].type = pii.id;
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
