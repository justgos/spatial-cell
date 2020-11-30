Shader "Custom/InstancedMetabolicParticles"
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
        uniform StructuredBuffer<MetabolicParticle> particles;
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
            MetabolicParticle p = particles[unity_InstanceID];

            /*float rotation = data.w * data.w * _Time.y * 0.5f;
            rotate2D(data.xz, rotation);*/

            unity_ObjectToWorld._11_21_31_41 = float4(meshScale / 10 * scale, 0, 0, 0);
            unity_ObjectToWorld._12_22_32_42 = float4(0, meshScale / 10 * scale, 0, 0);
            unity_ObjectToWorld._13_23_33_43 = float4(0, 0, meshScale / 10 * scale, 0);
            unity_ObjectToWorld._14_24_34_44 = float4(p.pos.xyz * scale, 1);
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
            MetabolicParticle p = particles[unity_InstanceID];
            /*float4 pos = mul(baseTransform,
                float4(transform_vector(p.pos, p.rot) * scale, 1)
            );*/

            v.vertex.xyz *= min(max(p.metabolites[0] / 10.0, 0.0), 1.0);

            v.vertex.xyz = transform_vector(v.vertex.xyz, p.rot);
            v.normal.xyz = transform_vector(v.normal.xyz, p.rot);
            float4 pos = mul(baseTransform,
                float4(p.pos * scale, 1)
            );
            float3 clippedCameraPos = float3(
                min(max(_WorldSpaceCameraPos.x, 0.0), simSize * scale),
                min(max(_WorldSpaceCameraPos.y, 0.0), simSize * scale),
                min(max(_WorldSpaceCameraPos.z, 0.0), simSize * scale)
            );
            float3 cameraDist = pos - clippedCameraPos;

            o.col = colormap[(uint)p.type % colormapLength];
            if(p.type == 0)
                o.col = float4(0.2, 0.2, 0.2, 1);
            o.col = float4(1, 0, 0, 1);
            o.col.rgb /= (1.0 + (abs(cameraDist.x) + abs(cameraDist.y) + abs(cameraDist.z)) / simSize / scale);


            if (
                !(p.flags & PARTICLE_FLAG_ACTIVE)
                || p.pos.x < visibleMinX
                || p.pos.x > visibleMaxX
                || p.pos.y < visibleMinY
                || p.pos.y > visibleMaxY
                || p.pos.z < visibleMinZ
                || p.pos.z > visibleMaxZ
            ) {
                v.vertex = 0;
                //o.col = float4(1, 0, 0, 1);
            }
#else
            o.col = float4(1, 1, 1, 1);
#endif
        }

        void surf (Input IN, inout SurfaceOutput o)
        {
            // Albedo comes from a texture tinted by color
            fixed4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
            o.Albedo = c.rgb * IN.col;
//#ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
//            Particle p = particles[unity_InstanceID];
//            o.Albedo.r = o.Albedo.r * (p.type % 2);
//#endif
            // Metallic and smoothness come from slider variables
            /*o.Metallic = _Metallic;
            o.Smoothness = _Glossiness;*/
            o.Alpha = c.a;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
