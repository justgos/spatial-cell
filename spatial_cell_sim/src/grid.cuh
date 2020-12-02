#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
//#include <cub/device/device_radix_sort.cuh>

#include "constants.cuh"
#include "macros.cuh"
#include "memory.cuh"

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

template <typename T>
__device__ __inline__ unsigned int
makeIdx(T p)
{
    return makeIdx(getGridIdx(p.pos.x), getGridIdx(p.pos.y), getGridIdx(p.pos.z));
}

template <typename T>
__global__ void
updateIndices(const T* curParticles, unsigned int* indices, const int numParticles)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numParticles)
        return;

    T p = curParticles[idx];
    indices[idx] = (p.flags & PARTICLE_FLAG_ACTIVE) ? makeIdx<T>(p) : MAX_GRID_INDEX;
}

__global__ void
updateGridRanges(const unsigned int* indices, unsigned int* gridRanges, const int numParticles)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= d_Config.gridSize)
        return;

    const int startIndex = (int)(idx * (float)numParticles / d_Config.gridSize);
    const int endIndex = min((int)((idx + 1) * (float)numParticles / d_Config.gridSize), numParticles);

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
    if (endIndex >= numParticles || indices[endIndex] != lastIdx)
        gridRanges[lastIdx * 2 + 1] = endIndex;
}


template <typename T>
class RadixSortPairs {
public:
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    RadixSortPairs(
        DeviceOnlyDoubleBuffer<unsigned int> *indices,
        DoubleBuffer<T> *items
    ) {
        /*
         * The fix for "uses too much shared data" is to change
         * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include\cub\device\dispatch\dispatch_radix_sort.cuh
         * reduce the number of threads per blocks from 512 to 384, in the `Policy700`, line 788 - the first of "Downsweep policies"
        */
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            indices->d_Current, indices->d_Next, items->d_Current, items->d_Next, indices->count);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        printf("Allocated %d bytes as temporary storage for RadixSortPairs\n", temp_storage_bytes);
    }

    void sort(
        DeviceOnlyDoubleBuffer<unsigned int>* indices,
        DoubleBuffer<T>* items
    ) {
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            indices->d_Current, indices->d_Next, items->d_Current, items->d_Next, indices->count);
    }

    ~RadixSortPairs() {
        cudaUnalloc(d_temp_storage);
    }
};


template <typename T>
void
updateGridAndSort(
    DoubleBuffer<T>* particles,
    DeviceOnlyDoubleBuffer<unsigned int>* indices,
    DeviceOnlySingleBuffer<unsigned int>* gridRanges,
    RadixSortPairs<T>* particleSort,
    const int numParticles,
    Config* config
) {
    updateIndices<T> KERNEL_ARGS2(CUDA_NUM_BLOCKS(numParticles), CUDA_THREADS_PER_BLOCK) (
        particles->d_Current,
        indices->d_Current,
        numParticles
    );
    //cudaDeviceSynchronize();
    particleSort->sort(indices, particles);
    //cudaDeviceSynchronize();
    particles->swap();
    indices->swap();
    //printCUDAIntArray(d_Indices, config.numParticles);
    gridRanges->clear();
    updateGridRanges KERNEL_ARGS2(CUDA_NUM_BLOCKS(config->gridSize), CUDA_THREADS_PER_BLOCK) (
        indices->d_Current,
        gridRanges->d_Current,
        numParticles
    );
    //cudaDeviceSynchronize();
}
