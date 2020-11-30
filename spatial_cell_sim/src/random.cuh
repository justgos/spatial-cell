#pragma once

#include <cuda_runtime.h>


#define scramble_float__mix(a,b,c) \
{ \
  a -= b; a -= c; a ^= (c>>13); \
  b -= c; b -= a; b ^= (a<<8); \
  c -= a; c -= b; c ^= (b>>13); \
  a -= b; a -= c; a ^= (c>>12);  \
  b -= c; b -= a; b ^= (a<<16); \
  c -= a; c -= b; c ^= (b>>5); \
  a -= b; a -= c; a ^= (c>>3);  \
  b -= c; b -= a; b ^= (a<<10); \
  c -= a; c -= b; c ^= (b>>15); \
}

// Ref: https://stackoverflow.com/a/6211738
/* transform float in [0,1] into a different float in [0,1] */
__device__ float
scramble_float(float f) {
    unsigned int magic1 = 0x96f563ae; /* number of your choice */
    unsigned int magic2 = 0xb93c7563; /* number of your choice */
    unsigned int j;
    j = reinterpret_cast<unsigned int&>(f);
    scramble_float__mix(magic1, magic2, j);
    return 2.3283064365386963e-10f * j;
}