#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "constants.cuh"

__device__ __inline__ unsigned int
getGridIdx(float coord)
{
    return min((unsigned int)(coord / d_Config.gridCellSize), d_Config.nGridCells - 1);
}

__device__ __inline__ unsigned int
makeIdx(unsigned int gridX, unsigned int gridY, unsigned int gridZ)
{
    return (gridX << (2 * d_Config.nGridCellsBits))
        | (gridY << d_Config.nGridCellsBits)
        | (gridZ << 0);
}

__device__ __inline__ unsigned int
makeIdx(Particle p)
{
    return makeIdx(getGridIdx(p.pos.x), getGridIdx(p.pos.y), getGridIdx(p.pos.z));
}

__global__ void
updateIndices(const Particle* curParticles, unsigned int* indices)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= d_Config.numParticles)
        return;

    Particle p = curParticles[idx];
    indices[idx] = (p.flags & PARTICLE_FLAG_ACTIVE) ? makeIdx(p) : MAX_GRID_INDEX;
}

__global__ void
updateGridRanges(const unsigned int* indices, unsigned int* gridRanges)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= d_Config.gridSize)
        return;

    const int startIndex = (int)(idx * (double)d_Config.numParticles / d_Config.gridSize);
    const int endIndex = min((int)((idx + 1) * (double)d_Config.numParticles / d_Config.gridSize), d_Config.numParticles);

    int lastIdx = indices[startIndex];
    if (startIndex <= 0 || indices[startIndex - 1] != lastIdx)
        gridRanges[lastIdx * 2] = startIndex;

    for (int i = startIndex; i < endIndex; i++) {
        int curIdx = indices[i];
        if (curIdx != lastIdx) {
            gridRanges[lastIdx * 2 + 1] = i;
            gridRanges[curIdx * 2] = i;
            lastIdx = curIdx;
        }
    }
    if (endIndex >= d_Config.numParticles || indices[endIndex] != lastIdx)
        gridRanges[lastIdx * 2 + 1] = endIndex;
}
