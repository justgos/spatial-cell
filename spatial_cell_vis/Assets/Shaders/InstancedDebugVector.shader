Shader "Custom/InstancedDebugVector"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        // Physically based Standard lighting model, and enable shadows on all light types
        #pragma surface surf Lambert noshadow vertex:vert
        #pragma instancing_options procedural:setup
        #pragma target 3.0

        #include "UnityCG.cginc"
        #include "Common.cginc"

        struct Input
        {
            float2 uv_MainTex;
            half4 col;
            UNITY_VERTEX_INPUT_INSTANCE_ID
        };

        sampler2D _MainTex;
#ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
        uniform StructuredBuffer<Particle> particles;
        uniform float4x4 baseTransform;
        uniform float meshScale;
        uniform float meshHeight;
        uniform float scale;
        uniform float simSize;
        uniform float visibleMinX;
        uniform float visibleMaxX;
        uniform float visibleMinY;
        uniform float visibleMaxY;
        uniform float visibleMinZ;
        uniform float visibleMaxZ;
        uniform int particleTypeFilter;
#endif
        fixed4 _Color;

        void setup()
        {
#ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
            Particle p = particles[unity_InstanceID];

            /*float rotation = data.w * data.w * _Time.y * 0.5f;
            rotate2D(data.xz, rotation);*/

            float radius = decodeLowUintToFloat16(p.r_pos_rot[0]);
            float3 pos = float3(
                decodeHighUintToFloat16(p.r_pos_rot[0]),
                decodeLowUintToFloat16(p.r_pos_rot[1]),
                decodeHighUintToFloat16(p.r_pos_rot[1])
            );

            unity_ObjectToWorld._11_21_31_41 = float4(meshScale / 0.005 * radius * 2 / 10 * scale, 0, 0, 0);
            unity_ObjectToWorld._12_22_32_42 = float4(0, meshScale / 0.005 * radius * 2 / 10 * scale, 0, 0);
            unity_ObjectToWorld._13_23_33_43 = float4(0, 0, meshScale / 0.005 * radius * 2 / 10 * scale, 0);
            unity_ObjectToWorld._14_24_34_44 = float4(pos.xyz * scale, 1);
            unity_WorldToObject = unity_ObjectToWorld;
            unity_WorldToObject._14_24_34 *= -1;
            unity_WorldToObject._11_22_33 = 1.0f / unity_WorldToObject._11_22_33;
#endif
        }

        void vert(inout appdata_full v, out Input o)
        {
            UNITY_INITIALIZE_OUTPUT(Input, o);
            UNITY_TRANSFER_INSTANCE_ID(v, o);
            //vertInstancingUVs(v.texcoord, o.uv_MainTex);

#ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
            Particle p = particles[unity_InstanceID];
            /*float4 pos = mul(baseTransform,
                float4(transform_vector(p.pos, p.rot) * scale, 1)
            );*/
            float4 debugVector = float4(
                decodeLowUintToFloat16(p.debugVector[0]),
                decodeHighUintToFloat16(p.debugVector[0]),
                decodeLowUintToFloat16(p.debugVector[1]),
                decodeHighUintToFloat16(p.debugVector[1])
            );
            if (abs(debugVector.x) < 1e-4 && abs(debugVector.y) < 1e-4 && abs(debugVector.z) < 1e-4) {
                v.vertex = 0;
                return;
            }
            float radius = decodeLowUintToFloat16(p.r_pos_rot[0]);
            float3 pos = float3(
                decodeHighUintToFloat16(p.r_pos_rot[0]),
                decodeLowUintToFloat16(p.r_pos_rot[1]),
                decodeHighUintToFloat16(p.r_pos_rot[1])
            );

            if (
                !(p.flags & PARTICLE_FLAG_ACTIVE)
                || pos.x < visibleMinX
                || pos.x > visibleMaxX
                || pos.y < visibleMinY
                || pos.y > visibleMaxY
                || pos.z < visibleMinZ
                || pos.z > visibleMaxZ
            ) {
                v.vertex = 0;
                //o.col = float4(1, 0, 0, 1);
                return;
            }

            v.vertex.xyz *= 0.2;
            float3 relativeDebugVector = debugVector.xyz * scale * 50.0f;  //  -p.pos.xyz;
            float4 debugVectorOrientation = quaternionFromTo(float3(0, 1, 0), normalize(relativeDebugVector));
            if (v.vertex.y > 0)
                v.vertex.y *= (length(relativeDebugVector) / meshScale) / meshHeight / 10;
            //v.vertex.xyz = transform_vector(v.vertex.xyz, quaternion(float3(1, 0, 0), 1.59));
            v.vertex.xyz = transform_vector(v.vertex.xyz, debugVectorOrientation);
            //v.vertex.xyz += p.debugVector.xyz / meshScale;
            pos = mul(baseTransform,
                float4(pos * scale, 1)
            );
            float3 clippedCameraPos = float3(
                min(max(_WorldSpaceCameraPos.x, 0.0), simSize * scale),
                min(max(_WorldSpaceCameraPos.y, 0.0), simSize * scale),
                min(max(_WorldSpaceCameraPos.z, 0.0), simSize * scale)
            );
            float3 cameraDist = pos - clippedCameraPos;

            //o.col = colormap[p.type % colormapLength];
            o.col = float4(1, 0, 0, 1);
            /*o.col.xyz = relativeDebugVector.xyz / meshScale * 5;*/
            //o.col.xyz = debugVectorOrientation.xyz;
            //o.col = float4(1, 1, 1, 1);
            o.col.rgb /= (1.0 + (abs(cameraDist.x) + abs(cameraDist.y) + abs(cameraDist.z)) / simSize / scale);

            v.vertex = 0;
            if (!(p.flags & PARTICLE_FLAG_ACTIVE)) {
                v.vertex = 0;
                o.col = float4(1, 0, 0, 1);
            }
#else
            o.col = float4(1, 1, 1, 1);
#endif
        }

        void surf (Input IN, inout SurfaceOutput o)
        {
            fixed4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
            o.Albedo = c.rgb * IN.col;
            o.Alpha = c.a;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
