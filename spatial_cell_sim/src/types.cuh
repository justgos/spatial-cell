#pragma once

#include <ctime>
#include <ratio>
#include <chrono>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "./constants.cuh"


typedef std::chrono::high_resolution_clock::time_point time_point;

struct ParticleInteraction {
    int type;
    int partnerId;
};

struct Particle {
    int id;
    int type;
    int flags;
    float radius;
    float3 pos;
    float4 rot;
    float3 velocity;
    float4 angularVelocity;
    int nActiveInteractions;
    ParticleInteraction interactions[4];
    float4 debugVector;

    __device__ __host__ Particle(
        int id = 0,
        int type = 0,
        int flags = 0,
        float radius = 0.0025,
        float3 pos = VECTOR_ZERO,
        float4 rot = QUATERNION_IDENTITY,
        float3 velocity = VECTOR_ZERO,
        float4 angularVelocity = QUATERNION_IDENTITY
    ) : id(id),
        type(type),
        flags(flags | PARTICLE_FLAG_ACTIVE),
        radius(radius),
        pos(pos),
        rot(rot),
        velocity(velocity),
        angularVelocity(angularVelocity),
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
        float radius = 0.0025,
        float3 pos = VECTOR_ZERO,
        float4 rot = QUATERNION_IDENTITY,
        float3 velocity = VECTOR_ZERO,
        float4 angularVelocity = QUATERNION_IDENTITY
    ) : Particle(id, type, flags, radius, pos, rot, velocity, angularVelocity)
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
        flags(p.flags),
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

struct ParticleTypeInfo {
    std::string category;
    std::string name;
    float radius;

    ParticleTypeInfo() {
        //
    }

    ParticleTypeInfo(
        std::string category,
        std::string name,
        float radius
    ) : category(category),
        name(name),
        radius(radius)
    {
        //
    }
};
