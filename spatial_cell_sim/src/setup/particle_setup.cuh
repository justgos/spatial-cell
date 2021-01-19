#pragma once

#include <random>

#include <cuda_runtime.h>
#include <crt/math_functions.h>

#include "../types.cuh"
#include "../constants.cuh"
#include "../math.cuh"


template <typename T> void
fillParticlesUniform(
	int count,
	int type,
	T* h_Particles,
	int *h_nActiveParticles,
    std::unordered_map<int, ParticleTypeInfo> *particleTypeInfo,
    const Config *config,
    std::function<double()> rng
) {
    printf("fillParticlesUniform %d\n", count);
    for (int i = h_nActiveParticles[0]; i < h_nActiveParticles[0] + count; i++)
    {
        h_Particles[i] = T(
            i,
            type,
            0,
            (*particleTypeInfo)[type].radius,
            make_float3(
                rng() * config->simSize,
                rng() * config->simSize,
                rng() * config->simSize
            ),
            random_rotation_host(rng)
        );
    }
    h_nActiveParticles[0] += count;
}

template <typename T> void
fillParticlesStraightLine(
    std::vector<int>* types,
    float3 startPos,
    float3 dPos,
    T* h_Particles,
    int* h_nActiveParticles,
    std::unordered_map<int, ParticleTypeInfo>* particleTypeInfo,
    const Config* config,
    std::function<double()> rng
) {
    for (int i = h_nActiveParticles[0]; i < h_nActiveParticles[0] + types->size(); i++)
    {
        int type = (*types)[i - h_nActiveParticles[0]];
        float radius = (*particleTypeInfo)[type].radius;
        float3 pos = add(startPos, mul(dPos, i - h_nActiveParticles[0]));
        h_Particles[i] = T(
            i,
            type,
            0,
            radius,
            make_float3(
                min(max(pos.x, 0.0f), config->simSize),
                min(max(pos.y, 0.0f), config->simSize),
                min(max(pos.z, 0.0f), config->simSize)
            ),
            QUATERNION_IDENTITY
            //random_rotation_host(rng)
        );
    }
    h_nActiveParticles[0] += types->size();
}

template <typename T> void
fillParticlesWrappedChain(
    std::vector<int> *types,
    float3 startPos,
    T* h_Particles,
    int* h_nActiveParticles,
    std::unordered_map<int, ParticleTypeInfo>* particleTypeInfo,
    const Config* config,
    std::function<double()> rng
) {
    float3 pos = startPos;
    float4 rotation = random_rotation_host(rng);
    // Maximum orientation change per residue
    float wrapRate = 0.3;
    for (int i = h_nActiveParticles[0]; i < h_nActiveParticles[0] + types->size(); i++)
    {
        int type = (*types)[i - h_nActiveParticles[0]];
        float radius = (*particleTypeInfo)[type].radius;
        if (i > h_nActiveParticles[0]) {
            rotation = slerp(rotation, random_rotation_host(rng), wrapRate);
            float3 dPos = -((*particleTypeInfo)[type - 1].radius + radius) *
                transform_vector(VECTOR_UP, rotation);
            pos += dPos;
        }
        h_Particles[i] = T(
            i,
            type,
            0,
            radius,
            make_float3(
                min(max(pos.x, 0.0f), config->simSize),
                min(max(pos.y, 0.0f), config->simSize),
                min(max(pos.z, 0.0f), config->simSize)
            ),
            rotation
        );
    }
    h_nActiveParticles[0] += types->size();
}

template <typename T> void
fillParticlesPlane(
    int countSqrt,
    int type,
    float3 center,
    float3 normal,
    T* h_Particles,
    int* h_nActiveParticles,
    std::unordered_map<int, ParticleTypeInfo>* particleTypeInfo,
    const Config* config,
    std::function<double()> rng
) {
    float3 dir1 = cross(normal, VECTOR_UP);
    float3 dir2 = cross(normal, dir1);
    dir1 = mul(dir1, (*particleTypeInfo)[type].radius * 2.0);
    dir2 = mul(dir2, (*particleTypeInfo)[type].radius * 2.0);
    float3 startPos = add(
        center,
        add(
            negate(mul(dir1, countSqrt * 0.5)),
            negate(mul(dir2, countSqrt * 0.5))
        )
    );
    for (int i = h_nActiveParticles[0]; i < h_nActiveParticles[0] + countSqrt * countSqrt; i++)
    {
        float3 pos = add(
            startPos,
            add(
                mul(dir1, (i - h_nActiveParticles[0]) % countSqrt),
                mul(dir2, (i - h_nActiveParticles[0]) / countSqrt)
            )
        );
        h_Particles[i] = T(
            i,
            type,
            0,
            (*particleTypeInfo)[type].radius,
            make_float3(
                min(max(pos.x, 0.0f), config->simSize),
                min(max(pos.y, 0.0f), config->simSize),
                min(max(pos.z, 0.0f), config->simSize)
            ),
            //random_rotation_host(rng)
            quaternionFromTo(VECTOR_UP, normal)
        );
    }
    h_nActiveParticles[0] += countSqrt * countSqrt;
}

// Ref: https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html
std::vector<float3> fibonacci_spiral_sphere(const int num_points) {
    std::vector<float3> vectors;
    vectors.reserve(num_points);

    const double gr = (sqrt(5.0) + 1.0) / 2.0;  // golden ratio = 1.6180339887498948482
    const double ga = (2.0 - gr) * (2.0 * PI);  // golden angle = 2.39996322972865332

    for (size_t i = 1; i <= num_points; ++i) {
        const double lat = asin(-1.0 + 2.0 * double(i) / (num_points + 1));
        const double lon = ga * i;

        const double x = cos(lon) * cos(lat);
        const double y = sin(lon) * cos(lat);
        const double z = sin(lat);

        vectors.push_back(make_float3(x, y, z));
    }

    return vectors;
}

template <typename T> void
fillParticlesSphere(
    int count,
    int type,
    float3 center,
    T* h_Particles,
    int* h_nActiveParticles,
    std::unordered_map<int, ParticleTypeInfo>* particleTypeInfo,
    const Config* config,
    std::function<double()> rng
) {
    std::vector<float3> pointsOnTheSphere = fibonacci_spiral_sphere(count);
    float r = sqrt(count / 1000.0 * pow((*particleTypeInfo)[type].radius / 2.5, 2.0)) * 40.0;
    for (int i = h_nActiveParticles[0]; i < h_nActiveParticles[0] + count; i++)
    {
        float3 pos = add(
            center,
            mul(pointsOnTheSphere[i - h_nActiveParticles[0]], r)
        );
        float3 posDirection = normalize(add(pos, negate(center)));
        float3 rotAxis = cross(VECTOR_UP, posDirection);
        h_Particles[i] = T(
            i,
            type,
            0,
            (*particleTypeInfo)[type].radius,
            make_float3(
                min(max(pos.x, 0.0f), config->simSize),
                min(max(pos.y, 0.0f), config->simSize),
                min(max(pos.z, 0.0f), config->simSize)
            ),
            //random_rotation_host(rng)
            quaternionFromTo(VECTOR_UP, posDirection)
            //quaternion(rotAxis, atan2(dot(posDirection, cross(VECTOR_UP, rotAxis)), dot(posDirection, VECTOR_UP)))
        );
    }
    h_nActiveParticles[0] += count;
}

template <typename T> void
instantiateComplex(
    int id,
    float3 center,
    T* h_Particles,
    int* h_nActiveParticles,
    std::unordered_map<int, ParticleTypeInfo>* particleTypeInfo,
    std::unordered_map<int, ComplexInfo> *complexInfo,
    const Config* config,
    std::function<double()> rng
) {
    ComplexInfo c = (*complexInfo)[id];
    for (int i = h_nActiveParticles[0]; i < h_nActiveParticles[0] + c.nParticipants; i++)
    {
        ComplexParticipantInfo cp = c.participants[i - h_nActiveParticles[0]];
        float3 pos = center + cp.position;
        h_Particles[i] = T(
            i,
            cp.type,
            0,
            (*particleTypeInfo)[cp.type].radius,
            make_float3(
                min(max(pos.x, 0.0f), config->simSize),
                min(max(pos.y, 0.0f), config->simSize),
                min(max(pos.z, 0.0f), config->simSize)
            ),
            cp.rotation
        );
    }
    h_nActiveParticles[0] += c.nParticipants;
}
