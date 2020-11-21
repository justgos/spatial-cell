#pragma once

#include "types.cuh"


#define PI 3.1415926535f
#define VECTOR_ZERO make_float3(0, 0, 0)
#define VECTOR_UP make_float3(0, 1, 0)
#define VECTOR_RIGHT make_float3(1, 0, 0)
#define VECTOR_FORWARD make_float3(0, 0, 1)

#define QUATERNION_IDENTITY make_float4(0, 0, 0, 1)

#define MAX_GRID_INDEX 0xFFFFFFFF

#define PARTICLE_FLAG_ACTIVE 0x0001

__constant__ Config d_Config;
