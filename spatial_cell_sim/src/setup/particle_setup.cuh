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
    int count,
    int type,
    float3 startPos,
    float3 dPos,
    T* h_Particles,
    int* h_nActiveParticles,
    const Config* config,
    std::function<double()> rng
) {
    for (int i = h_nActiveParticles[0]; i < h_nActiveParticles[0] + count; i++)
    {
        float3 pos = add(startPos, mul(dPos, i - h_nActiveParticles[0]));
        h_Particles[i] = T(
            i,
            type,
            0,
            make_float3(
                min(max(pos.x, 0.0f), 1.0f),
                min(max(pos.y, 0.0f), 1.0f),
                min(max(pos.z, 0.0f), 1.0f)
            ),
            random_rotation_host(rng)
        );
    }
    h_nActiveParticles[0] += count;
}

template <typename T> void
fillParticlesPlane(
    int countSqrt,
    int type,
    float3 center,
    float3 normal,
    T* h_Particles,
    int* h_nActiveParticles,
    const Config* config,
    std::function<double()> rng
) {
    float3 dir1 = cross(normal, VECTOR_UP);
    float3 dir2 = cross(normal, dir1);
    dir1 = mul(dir1, 0.005);
    dir2 = mul(dir2, 0.005);
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
            make_float3(
                min(max(pos.x, 0.0f), 1.0f),
                min(max(pos.y, 0.0f), 1.0f),
                min(max(pos.z, 0.0f), 1.0f)
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
    const Config* config,
    std::function<double()> rng
) {
    std::vector<float3> pointsOnTheSphere = fibonacci_spiral_sphere(count);
    float r = sqrt(count / 800.0) * 0.04;
    for (int i = h_nActiveParticles[0]; i < h_nActiveParticles[0] + count; i++)
    {
        float3 pos = add(
            center,
            mul(pointsOnTheSphere[i - h_nActiveParticles[0]], r)
        );
        float3 posDirection = normalized(add(pos, negate(center)));
        float3 rotAxis = cross(VECTOR_UP, posDirection);
        h_Particles[i] = T(
            i,
            type,
            0,
            make_float3(
                min(max(pos.x, 0.0f), 1.0f),
                min(max(pos.y, 0.0f), 1.0f),
                min(max(pos.z, 0.0f), 1.0f)
            ),
            //random_rotation_host(rng)
            quaternionFromTo(VECTOR_UP, posDirection)
            //quaternion(rotAxis, atan2(dot(posDirection, cross(VECTOR_UP, rotAxis)), dot(posDirection, VECTOR_UP)))
        );
    }
    h_nActiveParticles[0] += count;
}
