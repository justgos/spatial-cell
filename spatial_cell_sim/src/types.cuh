#pragma once

#include <ctime>
#include <ratio>
#include <chrono>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "./constants.cuh"


typedef std::chrono::high_resolution_clock::time_point time_point;

struct MinimalParticleTypeInfo {
    float radius;
    __int8 hydrophobic; // 0 - false, 1 - true, 2 - polar (e.g. lipid)

    __device__ __host__ MinimalParticleTypeInfo() {
        //
    }

    MinimalParticleTypeInfo(
        float radius,
        __int8 hydrophobic
    ) : radius(radius),
        hydrophobic(hydrophobic)
    {
        //
    }
};

struct ParticleTypeInfo : MinimalParticleTypeInfo {
    std::string category;
    std::string name;

    ParticleTypeInfo() {
        //
    }

    ParticleTypeInfo(
        std::string category,
        std::string name,
        float radius,
        __int8 hydrophobic
    ) : MinimalParticleTypeInfo(radius, hydrophobic),
        category(category),
        name(name)
    {
        //
    }
};

struct ParticleInteractionInfo {
    int id;
    int group;
    int firstPartnerType;
    __int8 firstPartnerState;
    int secondPartnerType;
    __int8 secondPartnerState;
    bool propagateState;
    bool setState;  // Should we change the participants's state to the ones specified in the interaction?
    bool waitForState;  // Should we wait for the participants' states to become same as specified before transitioning?
    bool waitForAlignment;
    bool onlyViaTransition;  // Can be initiated only via transition from another interaction
    int transitionTo;  // Upon being formed, this interaction might transform into another one
    /*
    * Upon transitioning to this interaction, another interaction should be enacted, polymerizing a new monomer.
    * The target interaction's `partnerType` (the other partner should be one of those in this interaction) 
    * treated as a type of monomer to by created.
    * If there's already an interaction with a target interaction's `group`, the new monomer will be attached
    * to that interaction's partner.
    */
    int polymerize;
    bool breakAfterTransition;  // This interaction is meant to change the participants' states, align 'em and then disintegrate
    float4 relativeOrientation;
    float3 relativePosition;

    __device__ __host__ ParticleInteractionInfo() {
        //
    }

    ParticleInteractionInfo(
        int id,
        int group,
        int firstPartnerType,
        __int8 firstPartnerState,
        int secondPartnerType,
        __int8 secondPartnerState,
        bool propagateState,
        bool setState,
        bool waitForState,
        bool waitForAlignment,
        bool onlyViaTransition,
        int transitionTo,
        int polymerize,
        bool breakAfterTransition,
        float4 relativeOrientation,
        float3 relativePosition
    ) : id(id),
        group(group),
        firstPartnerType(firstPartnerType),
        firstPartnerState(firstPartnerState),
        secondPartnerType(secondPartnerType),
        secondPartnerState(secondPartnerState),
        propagateState(propagateState),
        setState(setState),
        waitForState(waitForState),
        waitForAlignment(waitForAlignment),
        onlyViaTransition(onlyViaTransition),
        transitionTo(transitionTo),
        polymerize(polymerize),
        breakAfterTransition(breakAfterTransition),
        relativeOrientation(relativeOrientation),
        relativePosition(relativePosition)
    {
        //
    }
};

struct ComplexParticipantInfo {
    int type;
    float3 position;
    float4 rotation;

    ComplexParticipantInfo() {
        //
    }

    ComplexParticipantInfo(
        int type,
        float3 position,
        float4 rotation
    ) : type(type),
        position(position),
        rotation(rotation)
    {
        //
    }
};

struct ComplexInfo {
    int nParticipants;
    ComplexParticipantInfo* participants;

    ComplexInfo() {
        //
    }

    ComplexInfo(
        int nParticipants,
        ComplexParticipantInfo* participants
    ) : nParticipants(nParticipants),
        participants(participants)
    {
        //
    }
};

struct ParticleInteraction {
    int type;
    int group;
    int partnerId;
};

struct Particle {
    int id;
    int type;
    int flags;
    __int8 state;
    int lastStateChangeStep;
    float radius;
    __int8 hydrophobic; // 0 - false, 1 - true, 2 - polar (e.g. lipid)
    float3 pos;
    float4 rot;
    float3 velocity;
    float3 posNoise;
    float4 angularVelocity;
    float4 angularNoise;
    int nActiveInteractions;
    ParticleInteraction interactions[MAX_ACTIVE_INTERACTIONS];
    float4 debugVector;

    __device__ __host__ Particle(
        int id = 0,
        int type = 0,
        int flags = 0,
        float radius = 2.5,
        __int8 hydrophobic = HYDROPHYLIC,
        float3 pos = VECTOR_ZERO,
        float4 rot = QUATERNION_IDENTITY,
        float3 velocity = VECTOR_ZERO,
        float4 angularVelocity = QUATERNION_IDENTITY
    ) : id(id),
        type(type),
        flags(flags | PARTICLE_FLAG_ACTIVE),
        state(0),
        lastStateChangeStep(-1),
        radius(radius),
        hydrophobic(hydrophobic),
        pos(pos),
        rot(rot),
        velocity(velocity),
        posNoise(VECTOR_ZERO),
        angularVelocity(angularVelocity),
        angularNoise(QUATERNION_IDENTITY),
        nActiveInteractions(0),
        interactions(),
        debugVector(make_float4(0, 0, 0, 0))
    {
        //
    }
};

struct MetabolicParticle : Particle {
    float metabolites[NUM_METABOLITES];

    //using Particle::Particle;
    __device__ __host__ MetabolicParticle(
        int id = 0,
        int type = 0,
        int flags = 0,
        float radius = 2.5,
        __int8 hydrophobic = HYDROPHYLIC,
        float3 pos = VECTOR_ZERO,
        float4 rot = QUATERNION_IDENTITY,
        float3 velocity = VECTOR_ZERO,
        float4 angularVelocity = QUATERNION_IDENTITY
    ) : Particle(id, type, flags, radius, hydrophobic, pos, rot, velocity, angularVelocity)
    {
        memset(metabolites, 0, NUM_METABOLITES * sizeof(float));
    }
};

struct ReducedParticle {
    int id;
    int type;
    int flags;
    /*float3 pos;
    float4 rot;*/
    __half radius;
    __half posX;
    __half posY;
    __half posZ;
    __half rotX;
    __half rotY;
    __half rotZ;
    __half rotW;
    //float4 debugVector;
    __half debugVectorX;
    __half debugVectorY;
    __half debugVectorZ;
    __half debugVectorW;

    __device__ __host__ ReducedParticle(
        Particle p
    ) : id(p.id),
        type(p.type),
        flags(p.flags | (((int)p.state & 0x0000000F) << 16)),  // Pack a bit of state here for debugging
        radius(p.radius),
        posX(p.pos.x),
        posY(p.pos.y),
        posZ(p.pos.z),
        rotX(p.rot.x),
        rotY(p.rot.y),
        rotZ(p.rot.z),
        rotW(p.rot.w),
        debugVectorX(p.debugVector.x),
        debugVectorY(p.debugVector.y),
        debugVectorZ(p.debugVector.z),
        debugVectorW(p.debugVector.w)
        /*pos(p.pos),
        rot(p.rot)*/
        //debugVector(p.debugVector)
    {
        //
    }
};

struct ReducedMetabolicParticle : ReducedParticle {
    __half metabolites[REDUCED_NUM_METABOLITES];

    //using Particle::Particle;
    __device__ __host__ ReducedMetabolicParticle(
        MetabolicParticle p
    ) : ReducedParticle(p)
    {
        for (int i = 0; i < REDUCED_NUM_METABOLITES; i++)
            metabolites[i] = p.metabolites[i];
    }
};
