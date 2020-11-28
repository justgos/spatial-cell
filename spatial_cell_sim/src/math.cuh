#pragma once

#include <functional>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "constants.cuh"


__device__ __host__ __inline__ float
normsq(float3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__device__ __host__ __inline__ float
normsq(float4 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
}

__device__ __host__ __inline__ float
norm(float3 a) {
    return sqrt(normsq(a));
}

__device__ __host__ __inline__ float3
normalized(float3 a) {
    float n = norm(a);
    return make_float3(
        a.x / n,
        a.y / n,
        a.z / n
    );
}

__device__ __host__ __inline__ float
dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ __inline__ float
dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __host__ __inline__ float3
cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __host__ __inline__ float3
add(float3 a, float3 b) {
    return make_float3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

__device__ __host__ __inline__ float3
mul(float3 a, float b) {
    return make_float3(
        a.x * b,
        a.y * b,
        a.z * b
    );
}

__device__ __host__ __inline__ float4
mul(float4 a, float4 b) {
    float cx = a.y * b.z - a.z * b.y;
    float cy = a.z * b.x - a.x * b.z;
    float cz = a.x * b.y - a.y * b.x;

    float dot = a.x * b.x + a.y * b.y + a.z * b.z;

    return make_float4(
        a.x * b.w + b.x * a.w + cx,
        a.y * b.w + b.y * a.w + cy,
        a.z * b.w + b.z * a.w + cz,
        a.w * b.w - dot
    );
}

__device__ __host__ __inline__ float
angle(float3 a, float3 b) {
    return acos(dot(a, b) / (norm(a) * norm(b)));
}

__device__ __host__ __inline__ float
angle(float4 a) {
    return 2.0f * acos(a.w);
}

__device__ __host__ __inline__ float4
quaternion(float3 axis, float angle) {
    float sinAngle = sin(angle * 0.5);
    float cosAngle = cos(angle * 0.5);
    return make_float4(
        axis.x * sinAngle,
        axis.y * sinAngle,
        axis.z * sinAngle,
        cosAngle
    );
}

__device__ __host__ __inline__ float4
quaternionFromTo(float3 a, float3 b) {
    return quaternion(cross(a, b), angle(a, b));
}

__device__ __host__ __inline__ float3
negate(float3 a) {
    return make_float3(
        -a.x,
        -a.y,
        -a.z
    );
}

__device__ __host__ __inline__ float4
inverse(float4 a) {
    float invNorm = 1.0f / normsq(a);

    return make_float4(
        -a.x * invNorm,
        -a.y * invNorm,
        -a.z * invNorm,
        a.w * invNorm
    );
}

__device__ __host__ __inline__ float4
lerp(float4 a, float4 b, float amount) {
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

__device__ __host__ __inline__ float4
slerp(float4 a, float4 b, float amount) {
    const float epsilon = 1e-6f;

    float t = amount;

    float cosOmega = a.x * b.x + a.y * b.y +
        a.z * b.z + a.w * b.w;

    bool flip = false;

    if (cosOmega < 0.0f)
    {
        flip = true;
        cosOmega = -cosOmega;
    }

    float s1, s2;

    if (cosOmega > (1.0f - epsilon))
    {
        // Too close, do straight linear interpolation.
        s1 = 1.0f - t;
        s2 = (flip) ? -t : t;
    }
    else
    {
        float omega = acos(cosOmega);
        float invSinOmega = (1 / sin(omega));

        s1 = sin((1.0f - t) * omega) * invSinOmega;
        s2 = (flip)
            ? -sin(t * omega) * invSinOmega
            : sin(t * omega) * invSinOmega;
    }

    return make_float4(
        s1 * a.x + s2 * b.x,
        s1 * a.y + s2 * b.y,
        s1 * a.z + s2 * b.z,
        s1 * a.w + s2 * b.w
    );
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

float4
random_rotation_host(std::function<double()> rng) {
    float u = rng(),
        v = rng(),
        w = rng();
    float su = sqrt(u),
        su1 = sqrt(1 - u);
    return make_float4(
        su1 * sin(2 * PI * v),
        su1 * cos(2 * PI * v),
        su * sin(2 * PI * w),
        su * cos(2 * PI * w)
    );
}

__device__ __host__ __inline__ float3
transform_vector(float3 a, float4 q) {
    /*float3 u = make_float3(q.x, q.y, q.z);
    float s = q.w;
    return add(
        mul(u, dot(u, a) * 2),
        add(
            mul(a, s * s - dot(u, u)),
            mul(cross(u, a), s * 2)
        )
    );*/
    float x2 = q.x + q.x;
    float y2 = q.y + q.y;
    float z2 = q.z + q.z;

    float wx2 = q.w * x2;
    float wy2 = q.w * y2;
    float wz2 = q.w * z2;
    float xx2 = q.x * x2;
    float xy2 = q.x * y2;
    float xz2 = q.x * z2;
    float yy2 = q.y * y2;
    float yz2 = q.y * z2;
    float zz2 = q.z * z2;

    return make_float3(
        a.x * (1.0f - yy2 - zz2) + a.y * (xy2 - wz2) + a.z * (xz2 + wy2),
        a.x * (xy2 + wz2) + a.y * (1.0f - xx2 - zz2) + a.z * (yz2 - wx2),
        a.x * (xz2 - wy2) + a.y * (yz2 + wx2) + a.z * (1.0f - xx2 - yy2)
    );
}
