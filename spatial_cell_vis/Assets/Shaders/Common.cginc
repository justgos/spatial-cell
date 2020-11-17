#define PARTICLE_FLAG_ACTIVE 0x0001

struct Particle {
	float3 pos;
    float __padding1;
	float4 rot;
	float3 velocity;
	int type;
    int flags;
    float3 __padding2;
};

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
