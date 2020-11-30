// cudaNoise
// Library of common 3D noise functions for CUDA kernels

#ifndef cudanoise_cuh
#define cudanoise_cuh

#include <cuda_runtime.h>

#include "util.cuh"

namespace cudaNoise {


// Device constants for noise

__device__ __constant__ float gradMap[12][3] = { { 1.0f, 1.0f, 0.0f },{ -1.0f, 1.0f, 0.0f },{ 1.0f, -1.0f, 0.0f },{ -1.0f, -1.0f, 0.0f },
{ 1.0f, 0.0f, 1.0f },{ -1.0f, 0.0f, 1.0f },{ 1.0f, 0.0f, -1.0f },{ -1.0f, 0.0f, -1.0f },
{ 0.0f, 1.0f, 1.0f },{ 0.0f, -1.0f, 1.0f },{ 0.0f, 1.0f, -1.0f },{ 0.0f, -1.0f, -1.0f } };


// Helper functions for noise

// Linearly interpolate between two float values
__device__ __forceinline__  float lerp(float a, float b, float ratio)
{
	return a * (1.0f - ratio) + b * ratio;
}

// 1D cubic interpolation with four points
__device__ __forceinline__ float cubic(float p0, float p1, float p2, float p3, float x)
{
	return p1 + 0.5f * x * (p2 - p0 + x * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + x * (3.0f * (p1 - p2) + p3 - p0)));
}

// Fast gradient function for gradient noise
__device__ __forceinline__ float grad(int hash, float x, float y, float z)
{
	switch (hash & 0xF)
	{
	case 0x0: return  x + y;
	case 0x1: return -x + y;
	case 0x2: return  x - y;
	case 0x3: return -x - y;
	case 0x4: return  x + z;
	case 0x5: return -x + z;
	case 0x6: return  x - z;
	case 0x7: return -x - z;
	case 0x8: return  y + z;
	case 0x9: return -y + z;
	case 0xA: return  y - z;
	case 0xB: return -y - z;
	case 0xC: return  y + x;
	case 0xD: return -y + z;
	case 0xE: return  y - x;
	case 0xF: return -y - z;
	default: return 0; // never happens
	}
}

// Ken Perlin's fade function for Perlin noise
__device__ __forceinline__ float fade(float t)
{
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);         // 6t^5 - 15t^4 + 10t^3
}

// Dot product using a float[3] and float parameters
// NOTE: could be cleaned up
__device__ __forceinline__ float dot(float g[3], float x, float y, float z) {
	return g[0] * x + g[1] * y + g[2] * z;
}

// Random value for simplex noise [0, 255]
__device__ __forceinline__ unsigned char calcPerm(int p)
{
	return (unsigned char)(hash(p));
}

// Random value for simplex noise [0, 11]
__device__ __forceinline__ unsigned char calcPerm12(int p)
{
	return (unsigned char)(hash(p) % 12);
}

// Noise functions

// Simplex noise adapted from Java code by Stefan Gustafson and Peter Eastman
__device__  float simplexNoise(float3 pos, float scale, int seed)
{
	float xin = pos.x * scale;
	float yin = pos.y * scale;
	float zin = pos.z * scale;

	// Skewing and unskewing factors for 3 dimensions
	float F3 = 1.0f / 3.0f;
	float G3 = 1.0f / 6.0f;

	float n0, n1, n2, n3; // Noise contributions from the four corners

							// Skew the input space to determine which simplex cell we're in
	float s = (xin + yin + zin)*F3; // Very nice and simple skew factor for 3D
	int i = floorf(xin + s);
	int j = floorf(yin + s);
	int k = floorf(zin + s);
	float t = (i + j + k)*G3;
	float X0 = i - t; // Unskew the cell origin back to (x,y,z) space
	float Y0 = j - t;
	float Z0 = k - t;
	float x0 = xin - X0; // The x,y,z distances from the cell origin
	float y0 = yin - Y0;
	float z0 = zin - Z0;

	// For the 3D case, the simplex shape is a slightly irregular tetrahedron.
	// Determine which simplex we are in.
	int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
	int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
	if (x0 >= y0) {
		if (y0 >= z0)
		{
			i1 = 1.0f; j1 = 0.0f; k1 = 0.0f; i2 = 1.0f; j2 = 1.0f; k2 = 0.0f;
		} // X Y Z order
		else if (x0 >= z0) { i1 = 1.0f; j1 = 0.0f; k1 = 0.0f; i2 = 1.0f; j2 = 0.0f; k2 = 1.0f; } // X Z Y order
		else { i1 = 0.0f; j1 = 0.0f; k1 = 1.0f; i2 = 1.0f; j2 = 0.0f; k2 = 1.0f; } // Z X Y order
	}
	else { // x0<y0
		if (y0 < z0) { i1 = 0.0f; j1 = 0.0f; k1 = 1.0f; i2 = 0.0f; j2 = 1; k2 = 1.0f; } // Z Y X order
		else if (x0 < z0) { i1 = 0.0f; j1 = 1.0f; k1 = 0.0f; i2 = 0.0f; j2 = 1.0f; k2 = 1.0f; } // Y Z X order
		else { i1 = 0.0f; j1 = 1.0f; k1 = 0.0f; i2 = 1.0f; j2 = 1.0f; k2 = 0.0f; } // Y X Z order
	}

	// A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
	// a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
	// a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
	// c = 1/6.
	float x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
	float y1 = y0 - j1 + G3;
	float z1 = z0 - k1 + G3;
	float x2 = x0 - i2 + 2.0f*G3; // Offsets for third corner in (x,y,z) coords
	float y2 = y0 - j2 + 2.0f*G3;
	float z2 = z0 - k2 + 2.0f*G3;
	float x3 = x0 - 1.0f + 3.0f*G3; // Offsets for last corner in (x,y,z) coords
	float y3 = y0 - 1.0f + 3.0f*G3;
	float z3 = z0 - 1.0f + 3.0f*G3;

	// Work out the hashed gradient indices of the four simplex corners
	int ii = i & 255;
	int jj = j & 255;
	int kk = k & 255;

	int gi0 = calcPerm12(seed + ii + calcPerm(seed + jj + calcPerm(seed + kk)));
	int gi1 = calcPerm12(seed + ii + i1 + calcPerm(seed + jj + j1 + calcPerm(seed + kk + k1)));
	int gi2 = calcPerm12(seed + ii + i2 + calcPerm(seed + jj + j2 + calcPerm(seed + kk + k2)));
	int gi3 = calcPerm12(seed + ii + 1 + calcPerm(seed + jj + 1 + calcPerm(seed + kk + 1)));

	// Calculate the contribution from the four corners
	float t0 = 0.6f - x0*x0 - y0*y0 - z0*z0;
	if (t0 < 0.0f) n0 = 0.0f;
	else {
		t0 *= t0;
		n0 = t0 * t0 * dot(gradMap[gi0], x0, y0, z0);
	}
	float t1 = 0.6f - x1*x1 - y1*y1 - z1*z1;
	if (t1 < 0.0f) n1 = 0.0f;
	else {
		t1 *= t1;
		n1 = t1 * t1 * dot(gradMap[gi1], x1, y1, z1);
	}
	float t2 = 0.6f - x2*x2 - y2*y2 - z2*z2;
	if (t2 < 0.0f) n2 = 0.0f;
	else {
		t2 *= t2;
		n2 = t2 * t2 * dot(gradMap[gi2], x2, y2, z2);
	}
	float t3 = 0.6f - x3*x3 - y3*y3 - z3*z3;
	if (t3 < 0.0f) n3 = 0.0f;
	else {
		t3 *= t3;
		n3 = t3 * t3 * dot(gradMap[gi3], x3, y3, z3);
	}

	// Add contributions from each corner to get the final noise value.
	// The result is scaled to stay just inside [-1,1]
	return 32.0f*(n0 + n1 + n2 + n3);
}

__device__ __forceinline__ unsigned int hashUni(unsigned int x)
{
	const int m = 0xFFFFFFFF;
	return ((float)x + 1.0f)/((float)m + 2.0f);
}

__device__ __forceinline__ unsigned int hashExp(unsigned int x)
{
	// const int nLevels = 256;
	// pow(hash(seed) % nLevels
	const int m = 0xFFFFFFFF;
	const float scale = 1/log(0.5f*((float)m + 2.0f));
  return -scale * log(0.5f*(float)x + 1.0f) + 1.0f;
}

__device__  float simplexExpNoise(float3 pos, float scale, int seed)
{
	float xin = pos.x * scale;
	float yin = pos.y * scale;
	float zin = pos.z * scale;

	// Skewing and unskewing factors for 3 dimensions
	float F3 = 1.0f / 3.0f;
	float G3 = 1.0f / 6.0f;

	float n0, n1, n2, n3; // Noise contributions from the four corners

	// Skew the input space to determine which simplex cell we're in
	float s = (xin + yin + zin)*F3; // Very nice and simple skew factor for 3D
	int i = floorf(xin + s);
	int j = floorf(yin + s);
	int k = floorf(zin + s);
	float t = (i + j + k)*G3;
	float X0 = i - t; // Unskew the cell origin back to (x,y,z) space
	float Y0 = j - t;
	float Z0 = k - t;
	float x0 = xin - X0; // The x,y,z distances from the cell origin
	float y0 = yin - Y0;
	float z0 = zin - Z0;

	// For the 3D case, the simplex shape is a slightly irregular tetrahedron.
	// Determine which simplex we are in.
	int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
	int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
	if (x0 >= y0) {
		if (y0 >= z0)
		{
			i1 = 1.0f; j1 = 0.0f; k1 = 0.0f; i2 = 1.0f; j2 = 1.0f; k2 = 0.0f;
		} // X Y Z order
		else if (x0 >= z0) { i1 = 1.0f; j1 = 0.0f; k1 = 0.0f; i2 = 1.0f; j2 = 0.0f; k2 = 1.0f; } // X Z Y order
		else { i1 = 0.0f; j1 = 0.0f; k1 = 1.0f; i2 = 1.0f; j2 = 0.0f; k2 = 1.0f; } // Z X Y order
	}
	else { // x0<y0
		if (y0 < z0) { i1 = 0.0f; j1 = 0.0f; k1 = 1.0f; i2 = 0.0f; j2 = 1; k2 = 1.0f; } // Z Y X order
		else if (x0 < z0) { i1 = 0.0f; j1 = 1.0f; k1 = 0.0f; i2 = 0.0f; j2 = 1.0f; k2 = 1.0f; } // Y Z X order
		else { i1 = 0.0f; j1 = 1.0f; k1 = 0.0f; i2 = 1.0f; j2 = 1.0f; k2 = 0.0f; } // Y X Z order
	}

	// A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
	// a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
	// a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
	// c = 1/6.
	float x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
	float y1 = y0 - j1 + G3;
	float z1 = z0 - k1 + G3;
	float x2 = x0 - i2 + 2.0f*G3; // Offsets for third corner in (x,y,z) coords
	float y2 = y0 - j2 + 2.0f*G3;
	float z2 = z0 - k2 + 2.0f*G3;
	float x3 = x0 - 1.0f + 3.0f*G3; // Offsets for last corner in (x,y,z) coords
	float y3 = y0 - 1.0f + 3.0f*G3;
	float z3 = z0 - 1.0f + 3.0f*G3;

	// Work out the hashed gradient indices of the four simplex corners
	int ii = i & 255;
	int jj = j & 255;
	int kk = k & 255;

	int gi0 = calcPerm12(seed + ii + calcPerm(seed + jj + calcPerm(seed + kk)));
	int gi1 = calcPerm12(seed + ii + i1 + calcPerm(seed + jj + j1 + calcPerm(seed + kk + k1)));
	int gi2 = calcPerm12(seed + ii + i2 + calcPerm(seed + jj + j2 + calcPerm(seed + kk + k2)));
	int gi3 = calcPerm12(seed + ii + 1 + calcPerm(seed + jj + 1 + calcPerm(seed + kk + 1)));

	// Calculate the contribution from the four corners
	float t0 = 0.6f - x0*x0 - y0*y0 - z0*z0;
	if (t0 < 0.0f) n0 = 0.0f;
	else {
		t0 *= t0;
		n0 = t0 * t0 * dot(gradMap[gi0], x0, y0, z0) * hashExp(gi0);
	}
	float t1 = 0.6f - x1*x1 - y1*y1 - z1*z1;
	if (t1 < 0.0f) n1 = 0.0f;
	else {
		t1 *= t1;
		n1 = t1 * t1 * dot(gradMap[gi1], x1, y1, z1) * hashExp(gi1);
	}
	float t2 = 0.6f - x2*x2 - y2*y2 - z2*z2;
	if (t2 < 0.0f) n2 = 0.0f;
	else {
		t2 *= t2;
		n2 = t2 * t2 * dot(gradMap[gi2], x2, y2, z2) * hashExp(gi2);
	}
	float t3 = 0.6f - x3*x3 - y3*y3 - z3*z3;
	if (t3 < 0.0f) n3 = 0.0f;
	else {
		t3 *= t3;
		n3 = t3 * t3 * dot(gradMap[gi3], x3, y3, z3) * hashExp(gi3);
	}

	// Add contributions from each corner to get the final noise value.
	// The result is scaled to stay just inside [-1,1]
	return 32.0f*(n0 + n1 + n2 + n3);
}

// Checker pattern
__device__  float checker(float3 pos, float scale, int seed)
{
	int ix = (int)(pos.x * scale);
	int iy = (int)(pos.y * scale);
	int iz = (int)(pos.z * scale);

	if ((ix + iy + iz) % 2 == 0)
		return 1.0f;

	return -1.0f;
}

// Random spots
__device__  float spots(float3 pos, float scale, int seed, float size, int minNum, int maxNum, float jitter, profileShape shape)
{
	if (size < EPSILON)
		return 0.0f;

	int ix = (int)(pos.x * scale);
	int iy = (int)(pos.y * scale);
	int iz = (int)(pos.z * scale);

	float u = pos.x - (float)ix;
	float v = pos.y - (float)iy;
	float w = pos.z - (float)iz;

	float val = -1.0f;

	// We need to traverse the entire 3x3x3 neighborhood in case there are spots in neighbors near the edges of the cell
	for (int x = -1; x < 2; x++)
	{
		for (int y = -1; y < 2; y++)
		{
			for (int z = -1; z < 2; z++)
			{
				int numSpots = randomIntRange(minNum, maxNum, seed + (ix + x) * 823746.0f + (iy + y) * 12306.0f + (iz + z) * 823452.0f + 3234874.0f);

				for (int i = 0; i < numSpots; i++)
				{
					float distU = u - x - (randomFloat(seed + (ix + x) * 23784.0f + (iy + y) * 9183.0f + (iz + z) * 23874.0f * i + 27432.0f) * jitter - jitter / 2.0f);
					float distV = v - y - (randomFloat(seed + (ix + x) * 12743.0f + (iy + y) * 45191.0f + (iz + z) * 144421.0f * i + 76671.0f) * jitter - jitter / 2.0f);
					float distW = w - z - (randomFloat(seed + (ix + x) * 82734.0f + (iy + y) * 900213.0f + (iz + z) * 443241.0f * i + 199823.0f) * jitter - jitter / 2.0f);

					float distanceSq = distU * distU + distV * distV + distW * distW;

					switch (shape)
					{
					case(SHAPE_STEP):
						if (distanceSq < size)
							val = fmaxf(val, 1.0f);
						else
							val = fmaxf(val, -1.0f);
						break;
					case(SHAPE_LINEAR):
						float distanceAbs = fabsf(distU) + fabsf(distV) + fabsf(distW);
						val = fmaxf(val, 1.0f - clamp(distanceAbs, 0.0f, size) / size);
						break;
					case(SHAPE_QUADRATIC):
						val = fmaxf(val, 1.0f - clamp(distanceSq, 0.0f, size) / size);
						break;
					}
				}
			}
		}
	}

	return val;
}

// Worley cellular noise
__device__  float worleyNoise(float3 pos, float scale, int seed, float size, int minNum, int maxNum, float jitter)
{
	if (size < EPSILON)
		return 0.0f;

	int ix = (int)(pos.x * scale);
	int iy = (int)(pos.y * scale);
	int iz = (int)(pos.z * scale);

	float u = pos.x - (float)ix;
	float v = pos.y - (float)iy;
	float w = pos.z - (float)iz;

	float minDist = 1000000.0f;

	// Traverse the whole 3x3 neighborhood looking for the closest feature point
	for (int x = -1; x < 2; x++)
	{
		for (int y = -1; y < 2; y++)
		{
			for (int z = -1; z < 2; z++)
			{
				int numPoints = randomIntRange(minNum, maxNum, seed + (ix + x) * 823746.0f + (iy + y) * 12306.0f + (iz + z) * 67262.0f);

				for (int i = 0; i < numPoints; i++)
				{
					float distU = u - x - (randomFloat(seed + (ix + x) * 23784.0f + (iy + y) * 9183.0f + (iz + z) * 23874.0f * i + 27432.0f) * jitter - jitter / 2.0f);
					float distV = v - y - (randomFloat(seed + (ix + x) * 12743.0f + (iy + y) * 45191.0f + (iz + z) * 144421.0f * i + 76671.0f) * jitter - jitter / 2.0f);
					float distW = w - z - (randomFloat(seed + (ix + x) * 82734.0f + (iy + y) * 900213.0f + (iz + z) * 443241.0f * i + 199823.0f) * jitter - jitter / 2.0f);

					float distanceSq = distU * distU + distV * distV + distW * distW;

					if (distanceSq < minDist)
						minDist = distanceSq;
				}
			}
		}
	}

	return __saturatef(minDist) * 2.0f - 1.0f;
}

// Tricubic interpolation
__device__  float tricubic(int x, int y, int z, float u, float v, float w)
{
	// interpolate along x first
	float x00 = cubic(randomGrid(x - 1, y - 1, z - 1), randomGrid(x, y - 1, z - 1), randomGrid(x + 1, y - 1, z - 1), randomGrid(x + 2, y - 1, z - 1), u);
	float x01 = cubic(randomGrid(x - 1, y - 1, z), randomGrid(x, y - 1, z), randomGrid(x + 1, y - 1, z), randomGrid(x + 2, y - 1, z), u);
	float x02 = cubic(randomGrid(x - 1, y - 1, z + 1), randomGrid(x, y - 1, z + 1), randomGrid(x + 1, y - 1, z + 1), randomGrid(x + 2, y - 1, z + 1), u);
	float x03 = cubic(randomGrid(x - 1, y - 1, z + 2), randomGrid(x, y - 1, z + 2), randomGrid(x + 1, y - 1, z + 2), randomGrid(x + 2, y - 1, z + 2), u);

	float x10 = cubic(randomGrid(x - 1, y, z - 1), randomGrid(x, y, z - 1), randomGrid(x + 1, y, z - 1), randomGrid(x + 2, y, z - 1), u);
	float x11 = cubic(randomGrid(x - 1, y, z), randomGrid(x, y, z), randomGrid(x + 1, y, z), randomGrid(x + 2, y, z), u);
	float x12 = cubic(randomGrid(x - 1, y, z + 1), randomGrid(x, y, z + 1), randomGrid(x + 1, y, z + 1), randomGrid(x + 2, y, z + 1), u);
	float x13 = cubic(randomGrid(x - 1, y, z + 2), randomGrid(x, y, z + 2), randomGrid(x + 1, y, z + 2), randomGrid(x + 2, y, z + 2), u);

	float x20 = cubic(randomGrid(x - 1, y + 1, z - 1), randomGrid(x, y + 1, z - 1), randomGrid(x + 1, y + 1, z - 1), randomGrid(x + 2, y + 1, z - 1), u);
	float x21 = cubic(randomGrid(x - 1, y + 1, z), randomGrid(x, y + 1, z), randomGrid(x + 1, y + 1, z), randomGrid(x + 2, y + 1, z), u);
	float x22 = cubic(randomGrid(x - 1, y + 1, z + 1), randomGrid(x, y + 1, z + 1), randomGrid(x + 1, y + 1, z + 1), randomGrid(x + 2, y + 1, z + 1), u);
	float x23 = cubic(randomGrid(x - 1, y + 1, z + 2), randomGrid(x, y + 1, z + 2), randomGrid(x + 1, y + 1, z + 2), randomGrid(x + 2, y + 1, z + 2), u);

	float x30 = cubic(randomGrid(x - 1, y + 2, z - 1), randomGrid(x, y + 2, z - 1), randomGrid(x + 1, y + 2, z - 1), randomGrid(x + 2, y + 2, z - 1), u);
	float x31 = cubic(randomGrid(x - 1, y + 2, z), randomGrid(x, y + 2, z), randomGrid(x + 1, y + 2, z), randomGrid(x + 2, y + 2, z), u);
	float x32 = cubic(randomGrid(x - 1, y + 2, z + 1), randomGrid(x, y + 2, z + 1), randomGrid(x + 1, y + 2, z + 1), randomGrid(x + 2, y + 2, z + 1), u);
	float x33 = cubic(randomGrid(x - 1, y + 2, z + 2), randomGrid(x, y + 2, z + 2), randomGrid(x + 1, y + 2, z + 2), randomGrid(x + 2, y + 2, z + 2), u);

	// interpolate along y
	float y0 = cubic(x00, x10, x20, x30, v);
	float y1 = cubic(x01, x11, x21, x31, v);
	float y2 = cubic(x02, x12, x22, x32, v);
	float y3 = cubic(x03, x13, x23, x33, v);

	// interpolate along z
	return cubic(y0, y1, y2, y3, w);
}

// Discrete noise (nearest neighbor)
__device__  float discreteNoise(float3 pos, float scale, int seed)
{
	int ix = (int)(pos.x * scale);
	int iy = (int)(pos.y * scale);
	int iz = (int)(pos.z * scale);

	return randomGrid(ix, iy, iz, seed);
}

// Linear value noise
__device__  float linearValue(float3 pos, float scale, int seed)
{
	int ix = (int)pos.x;
	int iy = (int)pos.y;
	int iz = (int)pos.z;

	float u = pos.x - ix;
	float v = pos.y - iy;
	float w = pos.z - iz;

	// Corner values
	float a000 = randomGrid(ix, iy, iz, seed);
	float a100 = randomGrid(ix + 1, iy, iz, seed);
	float a010 = randomGrid(ix, iy + 1, iz, seed);
	float a110 = randomGrid(ix + 1, iy + 1, iz, seed);
	float a001 = randomGrid(ix, iy, iz + 1, seed);
	float a101 = randomGrid(ix + 1, iy, iz + 1, seed);
	float a011 = randomGrid(ix, iy + 1, iz + 1, seed);
	float a111 = randomGrid(ix + 1, iy + 1, iz + 1, seed);

	// Linear interpolation
	float x00 = lerp(a000, a100, u);
	float x10 = lerp(a010, a110, u);
	float x01 = lerp(a001, a101, u);
	float x11 = lerp(a011, a111, u);

	float y0 = lerp(x00, x10, v);
	float y1 = lerp(x01, x11, v);

	return lerp(y0, y1, w);
}

// Linear value noise smoothed with Perlin's fade function
__device__  float fadedValue(float3 pos, float scale, int seed)
{
	int ix = (int)(pos.x * scale);
	int iy = (int)(pos.y * scale);
	int iz = (int)(pos.z * scale);

	float u = fade(pos.x - ix);
	float v = fade(pos.y - iy);
	float w = fade(pos.z - iz);

	// Corner values
	float a000 = randomGrid(ix, iy, iz);
	float a100 = randomGrid(ix + 1, iy, iz);
	float a010 = randomGrid(ix, iy + 1, iz);
	float a110 = randomGrid(ix + 1, iy + 1, iz);
	float a001 = randomGrid(ix, iy, iz + 1);
	float a101 = randomGrid(ix + 1, iy, iz + 1);
	float a011 = randomGrid(ix, iy + 1, iz + 1);
	float a111 = randomGrid(ix + 1, iy + 1, iz + 1);

	// Linear interpolation
	float x00 = lerp(a000, a100, u);
	float x10 = lerp(a010, a110, u);
	float x01 = lerp(a001, a101, u);
	float x11 = lerp(a011, a111, u);

	float y0 = lerp(x00, x10, v);
	float y1 = lerp(x01, x11, v);

	return lerp(y0, y1, w) / 2.0f * 1.0f;
}

// Tricubic interpolated value noise
__device__  float cubicValue(float3 pos, float scale, int seed)
{
	pos.x = pos.x * scale;
	pos.y = pos.y * scale;
	pos.z = pos.z * scale;

	int ix = (int)pos.x;
	int iy = (int)pos.y;
	int iz = (int)pos.z;

	float u = pos.x - ix;
	float v = pos.y - iy;
	float w = pos.z - iz;

	return tricubic(ix, iy, iz, u, v, w);
}

// Perlin gradient noise
__device__  float perlinNoise(float3 pos, float scale, int seed)
{
	pos.x = pos.x * scale;
	pos.y = pos.y * scale;
	pos.z = pos.z * scale;

	// zero corner integer position
	int ix = (int)floorf(pos.x);
	int iy = (int)floorf(pos.y);
	int iz = (int)floorf(pos.z);

	// current position within unit cube
	pos.x -= floorf(pos.x);
	pos.y -= floorf(pos.y);
	pos.z -= floorf(pos.z);

	// adjust for fade
	float u = fade(pos.x);
	float v = fade(pos.y);
	float w = fade(pos.z);

	// influence values
	float i000 = grad(randomIntGrid(ix, iy, iz, seed), pos.x, pos.y, pos.z);
	float i100 = grad(randomIntGrid(ix + 1, iy, iz, seed), pos.x - 1.0f, pos.y, pos.z);
	float i010 = grad(randomIntGrid(ix, iy + 1, iz, seed), pos.x, pos.y - 1.0f, pos.z);
	float i110 = grad(randomIntGrid(ix + 1, iy + 1, iz, seed), pos.x - 1.0f, pos.y - 1.0f, pos.z);
	float i001 = grad(randomIntGrid(ix, iy, iz + 1, seed), pos.x, pos.y, pos.z - 1.0f);
	float i101 = grad(randomIntGrid(ix + 1, iy, iz + 1, seed), pos.x - 1.0f, pos.y, pos.z - 1.0f);
	float i011 = grad(randomIntGrid(ix, iy + 1, iz + 1, seed), pos.x, pos.y - 1.0f, pos.z - 1.0f);
	float i111 = grad(randomIntGrid(ix + 1, iy + 1, iz + 1, seed), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f);

	// interpolation
	float x00 = lerp(i000, i100, u);
	float x10 = lerp(i010, i110, u);
	float x01 = lerp(i001, i101, u);
	float x11 = lerp(i011, i111, u);

	float y0 = lerp(x00, x10, v);
	float y1 = lerp(x01, x11, v);

	float avg = lerp(y0, y1, w);

	return avg;
}

/*
 * Noise derivative, from http://www.iquilezles.org/www/articles/morenoise/morenoise.htm
 */
__device__ float4 perlinNoise_d( float3 pos, float scale, int seed )
{
		pos.x = pos.x * scale;
		pos.y = pos.y * scale;
		pos.z = pos.z * scale;

    //float3 p = floor(pos);
    //float3 w = fract(pos);

		// zero corner integer position
		int ix = (int)floorf(pos.x);
		int iy = (int)floorf(pos.y);
		int iz = (int)floorf(pos.z);

		// current position within unit cube
		pos.x -= floorf(pos.x);
		pos.y -= floorf(pos.y);
		pos.z -= floorf(pos.z);

		float3 w = pos;

    float3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    float3 du = 30.0*w*w*(w*(w-2.0)+1.0);

    // float a = myRandomMagic( p+vec3(0,0,0) );
    // float b = myRandomMagic( p+vec3(1,0,0) );
    // float c = myRandomMagic( p+vec3(0,1,0) );
    // float d = myRandomMagic( p+vec3(1,1,0) );
    // float e = myRandomMagic( p+vec3(0,0,1) );
    // float f = myRandomMagic( p+vec3(1,0,1) );
    // float g = myRandomMagic( p+vec3(0,1,1) );
    // float h = myRandomMagic( p+vec3(1,1,1) );

		float a = grad(randomIntGrid(ix, iy, iz, seed), pos.x, pos.y, pos.z);
		float b = grad(randomIntGrid(ix + 1, iy, iz, seed), pos.x - 1.0f, pos.y, pos.z);
		float c = grad(randomIntGrid(ix, iy + 1, iz, seed), pos.x, pos.y - 1.0f, pos.z);
		float d = grad(randomIntGrid(ix + 1, iy + 1, iz, seed), pos.x - 1.0f, pos.y - 1.0f, pos.z);
		float e = grad(randomIntGrid(ix, iy, iz + 1, seed), pos.x, pos.y, pos.z - 1.0f);
		float f = grad(randomIntGrid(ix + 1, iy, iz + 1, seed), pos.x - 1.0f, pos.y, pos.z - 1.0f);
		float g = grad(randomIntGrid(ix, iy + 1, iz + 1, seed), pos.x, pos.y - 1.0f, pos.z - 1.0f);
		float h = grad(randomIntGrid(ix + 1, iy + 1, iz + 1, seed), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f);

    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

		float3 derivative = 2.0f* du * make_float3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
										k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
										k3 + k6*u.x + k5*u.y + k7*u.x*u.y );
    return make_float4( -1.0f+2.0f*(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z),  derivative.x, derivative.y, derivative.z);
}

__device__  float perlinExpNoise(float3 pos, float scale, int seed)
{
	pos.x = pos.x * scale;
	pos.y = pos.y * scale;
	pos.z = pos.z * scale;

	// zero corner integer position
	int ix = (int)floorf(pos.x);
	int iy = (int)floorf(pos.y);
	int iz = (int)floorf(pos.z);

	// current position within unit cube
	pos.x -= floorf(pos.x);
	pos.y -= floorf(pos.y);
	pos.z -= floorf(pos.z);

	// adjust for fade
	float u = fade(pos.x);
	float v = fade(pos.y);
	float w = fade(pos.z);

	// influence values
	float i000 = grad(randomIntGrid(ix, iy, iz, seed), pos.x, pos.y, pos.z) * hashExp(randomIntGrid(ix, iy, iz, seed));
	float i100 = grad(randomIntGrid(ix + 1, iy, iz, seed), pos.x - 1.0f, pos.y, pos.z) * hashExp(randomIntGrid(ix + 1, iy, iz, seed));
	float i010 = grad(randomIntGrid(ix, iy + 1, iz, seed), pos.x, pos.y - 1.0f, pos.z) * hashExp(randomIntGrid(ix, iy + 1, iz, seed));
	float i110 = grad(randomIntGrid(ix + 1, iy + 1, iz, seed), pos.x - 1.0f, pos.y - 1.0f, pos.z) * hashExp(randomIntGrid(ix + 1, iy + 1, iz, seed));
	float i001 = grad(randomIntGrid(ix, iy, iz + 1, seed), pos.x, pos.y, pos.z - 1.0f) * hashExp(randomIntGrid(ix, iy, iz + 1, seed));
	float i101 = grad(randomIntGrid(ix + 1, iy, iz + 1, seed), pos.x - 1.0f, pos.y, pos.z - 1.0f) * hashExp(randomIntGrid(ix + 1, iy, iz + 1, seed));
	float i011 = grad(randomIntGrid(ix, iy + 1, iz + 1, seed), pos.x, pos.y - 1.0f, pos.z - 1.0f) * hashExp(randomIntGrid(ix, iy + 1, iz + 1, seed));
	float i111 = grad(randomIntGrid(ix + 1, iy + 1, iz + 1, seed), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f) * hashExp(randomIntGrid(ix + 1, iy + 1, iz + 1, seed));

	// interpolation
	float x00 = lerp(i000, i100, u);
	float x10 = lerp(i010, i110, u);
	float x01 = lerp(i001, i101, u);
	float x11 = lerp(i011, i111, u);

	float y0 = lerp(x00, x10, v);
	float y1 = lerp(x01, x11, v);

	float avg = lerp(y0, y1, w);

	return avg;
}

// Derived noise functions

// Fast function for fBm using perlin noise
__device__  float repeaterPerlin(float3 pos, float scale, int seed, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		acc += perlinNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
		scale *= lacunarity;
		amp *= decay;
	}

	return acc;
}

__device__  float repeaterPerlin_d(float3 pos, float scale, int seed, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;
	//float3 d = make_float3( 0.0f, 0.0f, 0.0f );
	float2 d = make_float2( 0.0f, 0.0f );

	for (int i = 0; i < n; i++)
	{
		float4 n = perlinNoise_d(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed);
		//d += make_float3( n.y, n.z, n.w );
		d += make_float2( n.y, n.z );
		acc += n.x * amp / (1.0f + dot(d, d));
		scale *= lacunarity;
		amp *= decay;
		//pos = make_float3(0.8f * pos.x - 0.6f * pos.y, 0.6f * pos.x + 0.8f * pos.y, pos.z / 2.0f) * 2.0f;
	}

	return -acc;
}

__device__  float repeaterHybrid(float3 pos, float scale, int seed, int n, float lacunarity, float decay, float offset)
{
	float acc = 0.0f;
	float amp = 1.0f;
	float w = 1.0f;
	float val;

	for (int i = 0; i < n; i++)
	{
		val = (perlinNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) + offset) * amp;
		acc += val * w;
		w *= val;
		scale *= lacunarity;
		amp *= decay;
	}

	return acc;
}

__device__  float repeaterPerlinExp(float3 pos, float scale, int seed, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		acc += perlinExpNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
		scale *= lacunarity;
		amp *= decay;
	}

	return acc;
}

// Fast function for fBm using perlin absolute noise
// Originally called "turbulence", this method takes the absolute value of each octave before adding
__device__  float repeaterPerlinAbs(float3 pos, float scale, int seed, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		acc += fabsf(perlinNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed)) * amp;
		scale *= lacunarity;
		amp *= decay;
	}

	// Map the noise back to the standard expected range [-1, 1]
	return mapToSigned(acc);
}

// Fast function for fBm using simplex noise
__device__  float repeaterSimplex(float3 pos, float scale, int seed, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		acc += simplexNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
		scale *= lacunarity;
		amp *= decay;
	}

	return acc;
}

__device__  float repeaterSimplexExp(float3 pos, float scale, int seed, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		acc += simplexExpNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
		scale *= lacunarity;
		amp *= decay;
	}

	return acc;
}

// Fast function for fBm using simplex absolute noise
__device__  float repeaterSimplexAbs(float3 pos, float scale, int seed, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		acc += fabsf(simplexNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed)) * amp * 0.35f;
		scale *= lacunarity;
		amp *= decay;
	}

	return mapToSigned(acc);
}

// Generic fBm repeater
// NOTE: about 10% slower than the dedicated repeater functions
__device__  float repeater(float3 pos, float scale, int seed, int n, float lacunarity, float decay, basisFunction basis)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		switch (basis)
		{
		case(BASIS_CHECKER):
			acc += checker(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
			break;
		case(BASIS_DISCRETE):
			acc += discreteNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
			break;
		case(BASIS_LINEARVALUE):
			acc += linearValue(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
			break;
		case(BASIS_FADEDVALUE):
			acc += fadedValue(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
			break;
		case(BASIS_CUBICVALUE):
			acc += cubicValue(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
			break;
		case(BASIS_PERLIN):
			acc += perlinNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
			break;
		case(BASIS_SIMPLEX):
			acc += simplexNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed) * amp;
			break;
		case(BASIS_WORLEY):
			acc += worleyNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed, 0.1f, 4, 4, 1.0f) * amp;
			break;
		case(BASIS_SPOTS):
			acc += spots(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed, 0.1f, 0, 4, 1.0f, SHAPE_LINEAR) * amp;
			break;
		}

		scale *= lacunarity;
		amp *= decay;
	}

	return acc;
}

// Fractal Simplex noise
// Unlike the repeater function, which calculates a fixed number of noise octaves, the fractal function continues until
// the feature size is smaller than one pixel
__device__  float fractalSimplex(float3 pos, float scale, int seed, float du, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;

	float rdu = 1.0f / du;

	for (int i = 0; i < n; i++)
	{
		acc += simplexNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed * (i + 1)) * amp;
		scale *= lacunarity;
		amp *= decay;

		if (scale > rdu)
			break;
	}

	return acc;
}

// Generic turbulence function
// Uses a first pass of noise to offset the input vectors for the second pass
__device__  float turbulence(float3 pos, float scaleIn, float scaleOut, int seed, float strength, basisFunction inFunc, basisFunction outFunc)
{
	switch (inFunc)
	{
	case(BASIS_CHECKER):
		pos.x += checker(pos, scaleIn, seed) * strength;
		pos.y += checker(pos, scaleIn, seed) * strength;
		pos.z += checker(pos, scaleIn, seed) * strength;
		break;
	case(BASIS_LINEARVALUE):
		pos.x += linearValue(pos, scaleIn, seed) * strength;
		pos.y += linearValue(pos, scaleIn, seed) * strength;
		pos.z += linearValue(pos, scaleIn, seed) * strength;
		break;
	case(BASIS_FADEDVALUE):
		pos.x += fadedValue(pos, scaleIn, seed) * strength;
		pos.y += fadedValue(pos, scaleIn, seed) * strength;
		pos.z += fadedValue(pos, scaleIn, seed) * strength;
		break;
	case(BASIS_CUBICVALUE):
		pos.x += cubicValue(pos, scaleIn, seed) * strength;
		pos.y += cubicValue(pos, scaleIn, seed) * strength;
		pos.z += cubicValue(pos, scaleIn, seed) * strength;
		break;
	case(BASIS_PERLIN):
		pos.x += perlinNoise(pos, scaleIn, seed) * strength;
		pos.y += perlinNoise(pos, scaleIn, seed) * strength;
		pos.z += perlinNoise(pos, scaleIn, seed) * strength;
		break;
	case(BASIS_SIMPLEX):
		pos.x += simplexNoise(pos, scaleIn, seed) * strength;
		pos.y += simplexNoise(pos, scaleIn, seed) * strength;
		pos.z += simplexNoise(pos, scaleIn, seed) * strength;
		break;
	case(BASIS_WORLEY):
		pos.x += worleyNoise(pos, scaleIn, seed, 1.0f, 4, 4, 1.0f) * strength;
		pos.y += worleyNoise(pos, scaleIn, seed, 1.0f, 4, 4, 1.0f) * strength;
		pos.z += worleyNoise(pos, scaleIn, seed, 1.0f, 4, 4, 1.0f) * strength;
		break;
	}

	switch (outFunc)
	{
	case(BASIS_CHECKER):
		return checker(pos, scaleOut, seed);
		break;
	case(BASIS_LINEARVALUE):
		return linearValue(pos, scaleOut, seed);
		break;
	case(BASIS_FADEDVALUE):
		return fadedValue(pos, scaleOut, seed);
		break;
	case(BASIS_CUBICVALUE):
		return cubicValue(pos, scaleOut, seed);
		break;
	case(BASIS_PERLIN):
		return perlinNoise(pos, scaleOut, seed);
		break;
	case(BASIS_SIMPLEX):
		return simplexNoise(pos, scaleIn, seed);
		break;
	case(BASIS_WORLEY):
		return worleyNoise(pos, scaleIn, seed, 1.0f, 4, 4, 1.0f);
		break;
	}

	return 0.0f;
}

// Turbulence using repeaters for the first and second pass
__device__  float repeaterTurbulence(float3 pos, float scaleIn, float scaleOut, int seed, float strength, int n, basisFunction basisIn, basisFunction basisOut)
{
	pos.x += (repeater(make_float3(pos.x, pos.y, pos.z), scaleIn, seed, n, 2.0f, 0.5f, basisIn)) * strength;

	return repeater(pos, scaleOut, seed, n, 2.0f, 0.75f, basisOut);
}

} // namespace

#endif
