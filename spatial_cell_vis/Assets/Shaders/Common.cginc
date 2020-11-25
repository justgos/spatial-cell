﻿#define PARTICLE_FLAG_ACTIVE 0x0001
// Upgrade NOTE: excluded shader from DX11, OpenGL ES 2.0 because it uses unsized arrays
#pragma exclude_renderers d3d11 gles

struct ParticleInteraction {
    int type;
    int partnerId;
};

struct Particle {
    int id;
    int type;
    int flags;
	float3 pos;
    float2 __padding1;
	float4 rot;
	float3 velocity;
    int nActiveInteractions;
    ParticleInteraction interactions[4];
    float4 debugVector;
};


float
angle(float3 a, float3 b) {
    return acos(dot(a, b) / (length(a) * length(b)));
}

float4
quaternion(float3 axis, float angle) {
    float sinAngle = sin(angle * 0.5);
    float cosAngle = cos(angle * 0.5);
    return float4(
        axis.x * sinAngle,
        axis.y * sinAngle,
        axis.z * sinAngle,
        cosAngle
    );
}

float4
quaternionFromTo(float3 a, float3 b) {
    return quaternion(cross(a, b), angle(a, b));
}

float3
transform_vector(float3 a, float4 q) {
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

    return float3(
        a.x * (1.0f - yy2 - zz2) + a.y * (xy2 - wz2) + a.z * (xz2 + wy2),
        a.x * (xy2 + wz2) + a.y * (1.0f - xx2 - zz2) + a.z * (yz2 - wx2),
        a.x * (xz2 - wy2) + a.y * (yz2 + wx2) + a.z * (1.0f - xx2 - yy2)
    );
}

static const int colormapLength = 12;
static const float4 colormap[] = {
    float4(0.74609375, 0.81640625, 0.828125, 1),
    float4(0.12109375, 0.46875, 0.703125, 1),
    float4(0.6953125, 0.87109375, 0.5390625, 1),
    float4(0.19921875, 0.625, 0.171875, 1),
    float4(0.98046875, 0.6015625, 0.59765625, 1),
    float4(0.88671875, 0.1015625, 0.109375, 1),
    float4(0.98828125, 0.74609375, 0.43359375, 1),
    float4(0.99609375, 0.49609375, 0, 1),
    float4(0.7890625, 0.6953125, 0.8359375, 1),
    float4(0.4140625, 0.23828125, 0.6015625, 1),
    float4(0.99609375, 0.99609375, 0.59765625, 1),
    float4(0.6941176470588235, 0.34901960784313724, 0.1568627450980392, 1)
};