#pragma once


#define PI 3.1415926535f
#define VECTOR_ZERO make_float3(0, 0, 0)
#define VECTOR_UP make_float3(0, 1, 0)
#define VECTOR_DOWN make_float3(0, -1, 0)
#define VECTOR_RIGHT make_float3(1, 0, 0)
#define VECTOR_FORWARD make_float3(0, 0, 1)

#define QUATERNION_IDENTITY make_float4(0, 0, 0, 1)

#define MAX_GRID_INDEX 0xFFFFFFFF

#define NUM_METABOLITES 10
#define REDUCED_NUM_METABOLITES 4

#define PARTICLE_FLAG_ACTIVE 0x0001

#define PARTICLE_TYPE_LIPID 0
#define PARTICLE_TYPE_DNA 1
#define PARTICLE_TYPE_SOME_PRODUCT 2
#define PARTICLE_TYPE_METABOLIC 1000

#define CUDA_THREADS_PER_BLOCK 256
#define CUDA_NUM_BLOCKS(nItems) (nItems + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK
