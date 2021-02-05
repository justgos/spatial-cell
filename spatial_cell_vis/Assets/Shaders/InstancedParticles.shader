Shader "Custom/InstancedParticles"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _Glossiness ("Smoothness", Range(0,1)) = 0.5
        _Metallic ("Metallic", Range(0,1)) = 0.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            // compile shader into multiple variants, with and without shadows
            // (we don't care about any lightmaps yet, so skip these variants)
            //#pragma multi_compile_fwdbase nolightmap nodirlightmap nodynlightmap novertexlight
            //#pragma surface surf Lambert noshadow vertex:vert
            #pragma instancing_options procedural:setup
            #pragma target 3.0

            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "Common.cginc"
            // shadow helper functions and macros
            #include "AutoLight.cginc"

            /*struct Input
            {
                float2 uv_MainTex;
                half4 col;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };*/
            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                float3 normal : NORMAL;

                UNITY_VERTEX_INPUT_INSTANCE_ID //Insert
            };
            struct v2f
            {
                float2 uv : TEXCOORD0;
                SHADOW_COORDS(1) // put shadows data into TEXCOORD1
                fixed3 ambient : COLOR1;
                float4 pos : SV_POSITION;
                half4 col : COLOR0;
                float3 normal : NORMAL;
                float4 worldPos : TEXCOORD2;
                float3 viewDir : TEXCOORD3;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            sampler2D _MainTex;
#if defined(UNITY_INSTANCING_ENABLED) || defined(UNITY_PROCEDURAL_INSTANCING_ENABLED) || defined(UNITY_STEREO_INSTANCING_ENABLED)
            uniform StructuredBuffer<Particle> particles;
            uniform float4x4 baseTransform;
            uniform float meshScale;
            uniform float scale;
            uniform float simSize;
            uniform float visibleMinX;
            uniform float visibleMaxX;
            uniform float visibleMinY;
            uniform float visibleMaxY;
            uniform float visibleMinZ;
            uniform float visibleMaxZ;
            uniform int particleTypeFilter;
            uniform int targetParticleId;
    #endif
            half _Glossiness;
            half _Metallic;
            fixed4 _Color;

            // Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
            // See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
            // #pragma instancing_options assumeuniformscaling
            //UNITY_INSTANCING_BUFFER_START(Props)
            //    // put more per-instance properties here
            //UNITY_INSTANCING_BUFFER_END(Props)

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

            v2f vert(appdata_base v)
            {
                UNITY_SETUP_INSTANCE_ID(v);
                v2f o;
                UNITY_TRANSFER_INSTANCE_ID(v, o);

#if defined(UNITY_INSTANCING_ENABLED) || defined(UNITY_PROCEDURAL_INSTANCING_ENABLED) || defined(UNITY_STEREO_INSTANCING_ENABLED)
                Particle p = particles[unity_InstanceID];

                float radius = decodeLowUintToFloat16(p.r_pos_rot[0]);
                float3 pos = float3(
                    decodeHighUintToFloat16(p.r_pos_rot[0]),
                    decodeLowUintToFloat16(p.r_pos_rot[1]),
                    decodeHighUintToFloat16(p.r_pos_rot[1])
                );
                float4 rot = float4(
                    decodeLowUintToFloat16(p.r_pos_rot[2]),
                    decodeHighUintToFloat16(p.r_pos_rot[2]),
                    decodeLowUintToFloat16(p.r_pos_rot[3]),
                    decodeHighUintToFloat16(p.r_pos_rot[3])
                );

                if (
                    !(p.flags & PARTICLE_FLAG_ACTIVE)
                    || pos.x < visibleMinX
                    || pos.x > visibleMaxX
                    || pos.y < visibleMinY
                    || pos.y > visibleMaxY
                    || pos.z < visibleMinZ
                    || pos.z > visibleMaxZ
                    || (p.type < 8 && !(particleTypeFilter & (0x0000001 << p.type)))
                ) {
                    v.vertex = 0;
                    //o.col = float4(1, 0, 0, 1);
                }

                /*o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.texcoord;
                o.worldPos = mul(UNITY_MATRIX_M, v.vertex);
                o.viewDir = WorldSpaceViewDir(v.vertex);
                o.normal = UnityObjectToWorldNormal(v.normal);
                half3 worldNormal = UnityObjectToWorldNormal(v.normal);
                half nl = max(0, dot(worldNormal, _WorldSpaceLightPos0.xyz));
                o.diff = nl * _LightColor0.rgb;
                o.ambient = ShadeSH9(half4(worldNormal, 1));*/

            

                v.vertex.xyz = transform_vector(v.vertex.xyz, rot);
                v.normal.xyz = transform_vector(v.normal.xyz, rot);
                pos = mul(baseTransform,
                    float4(pos * scale, 1)
                );

                float3 clippedCameraPos = float3(
                    min(max(_WorldSpaceCameraPos.x, 0.0), simSize * scale),
                    min(max(_WorldSpaceCameraPos.y, 0.0), simSize * scale),
                    min(max(_WorldSpaceCameraPos.z, 0.0), simSize * scale)
                );
                float3 cameraDist = pos - clippedCameraPos;

                o.col = colormap[(uint)p.type % colormapLength];
                if (p.type == 0)
                    o.col = float4(0.9, 0.8, 0.3, 1);
                
                int state = (p.flags >> 16) & 0x0F;
                if (state > 0) {
                    o.col.rgb = float3(0.0, 1.0, 0.0);
                }

                o.col.rgb /= (1.0 + (abs(cameraDist.x) + abs(cameraDist.y) + abs(cameraDist.z)) / simSize / scale);

                if (targetParticleId == p.id)
                    o.col = float4(1, 0, 0, 1);

                /*pos *= scale;
                pos += v.vertex.xyz * (meshScale / 0.005 * radius * 2 / 10 * scale);*/

                o.pos = UnityObjectToClipPos(v.vertex.xyz);
                o.uv = v.texcoord;
                o.worldPos = mul(UNITY_MATRIX_M, v.vertex.xyz);
                o.viewDir = WorldSpaceViewDir(float4(v.vertex.xyz, 1));
                o.normal = UnityObjectToWorldNormal(v.normal);
                half3 worldNormal = UnityObjectToWorldNormal(v.normal);
                half nl = max(0, dot(worldNormal, _WorldSpaceLightPos0.xyz));
                o.ambient = nl;  // ShadeSH9(half4(worldNormal, 1));
                //o.col.xyz = abs(worldNormal.xyz);
    #else
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.texcoord;
                o.worldPos = mul(UNITY_MATRIX_M, v.vertex);
                o.viewDir = WorldSpaceViewDir(v.vertex);
                o.normal = UnityObjectToWorldNormal(v.normal);
                half3 worldNormal = UnityObjectToWorldNormal(v.normal);
                half nl = max(0, dot(worldNormal, _WorldSpaceLightPos0.xyz));
                o.ambient = ShadeSH9(half4(worldNormal, 1));

                o.col = float4(1, 1, 1, 1);
    #endif
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(i);

                fixed4 col = tex2D(_MainTex, i.uv) * _Color * i.col;
                col.xyz = (0.4 + 0.6 * i.ambient) * col.xyz;
                //float3 lightDir = normalize(_WorldSpaceLightPos0.xyz);

                //// darken light's illumination with shadow, keep ambient intact
                //half3 h = normalize(lightDir + i.viewDir);
                //float nh = max(0, dot(i.normal, h));
                //float spec = pow(nh, 48.0);

                //fixed3 lighting = i.diff + _LightColor0.rgb * spec + i.ambient;
                //col.rgb *= lighting;
                //col.rgb *= i.diff + i.ambient;
                return col;
            }

            ENDCG
        }
    }
    FallBack "Diffuse"
}
