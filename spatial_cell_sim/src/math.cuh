#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "constants.cuh"


__device__ __inline__ float
dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __inline__ float
dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __inline__ float3
cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __inline__ float3
add(float3 a, float3 b) {
    return make_float3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

__device__ __inline__ float3
mul(float3 a, float b) {
    return make_float3(
        a.x * b,
        a.y * b,
        a.z * b
    );
}

__device__ __inline__ float4
mul(float4 a, float4 b) {
    return make_float4(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
        a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x
    );
}

__device__ __inline__ float4
lepr(float4 a, float4 b, float amount) {
    float t = amount;
    float t1 = 1.0f - t;

    float4 r;

    float dot = a.x * b.x + a.y * b.y +
        a.z * b.z + a.w * b.w;

    if (dot >= 0.0f)
    {
        r.x = t1 * a.x + t * b.x;
        r.y = t1 * a.y + t * b.y;
        r.z = t1 * a.z + t * b.z;
        r.w = t1 * a.w + t * b.w;
    }
    else
    {
        r.x = t1 * a.x - t * b.x;
        r.y = t1 * a.y - t * b.y;
        r.z = t1 * a.z - t * b.z;
        r.w = t1 * a.w - t * b.w;
    }

    // Normalize it.
    float ls = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
    float invNorm = 1.0f / (float)sqrt((double)ls);

    r.x *= invNorm;
    r.y *= invNorm;
    r.z *= invNorm;
    r.w *= invNorm;

    return r;
}

__device__ __inline__ float4
random_rotation(curandState* rngState) {
    float u = curand_uniform(rngState),
        v = curand_uniform(rngState),
        w = curand_uniform(rngState);
    float su = sqrt(u),
        su1 = sqrt(1 - u);
    return make_float4(
        su1 * sin(2 * PI * v),
        su1 * cos(2 * PI * v),
        su * sin(2 * PI * w),
        su * cos(2 * PI * w)
    );
}

__device__ __inline__ float3
transform_vector(float3 a, float4 q) {
    float3 u = make_float3(q.x, q.y, q.z);
    float s = q.w;
    return add(
        mul(u, dot(u, a) * 2),
        add(
            mul(a, s * s - dot(u, u)),
            mul(cross(u, a), s * 2)
        )
    );
}
