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

__device__ __inline__ float4
getTargetRelativeOrientation(Particle p, Particle tp, float4 targetRelativeOrientation) {
	float4 relativeOrientationDelta = p.id < tp.id
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
    float3 relaxedRelativePosition = p.id < tp.id
        ? negate(transform_vector(VECTOR_UP, mul(tp.rot, mul(inverse(targetRelativeOrientation), targetRelativePositionRotation))))
        : transform_vector(VECTOR_UP, mul(tp.rot, targetRelativePositionRotation));
    relaxedRelativePosition.x *= interactionDistance;
    relaxedRelativePosition.y *= interactionDistance;
    relaxedRelativePosition.z *= interactionDistance;
    relaxedRelativePosition.x += tp.pos.x;
    relaxedRelativePosition.y += tp.pos.y;
    relaxedRelativePosition.z += tp.pos.z;
    return relaxedRelativePosition;
}
