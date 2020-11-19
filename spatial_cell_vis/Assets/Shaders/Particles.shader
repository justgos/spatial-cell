Shader "Custom/Particles"
{
	Properties{
		_MainTex("Texture", 2D) = "white" {}
		[HDR] _Color("Tint", Color) = (0, 0, 0, 1)
	}

	SubShader{
		//the material is completely non-transparent and is rendered at the same time as the other opaque geometry
		//Tags{ "RenderType" = "Opaque" "Queue" = "Geometry" }
		//Tags{ "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" }
		Tags {"Queue" = "AlphaTest" "IgnoreProjector" = "True" "RenderType" = "TransparentCutout"}

		/*Cull Off
		ZWrite Off
		Lighting Off
		Fog{ Mode off }
		ColorMask RGB
		Blend SrcAlpha OneMinusSrcAlpha*/

		Pass{
			CGPROGRAM

			//include useful shader functions
			#include "UnityCG.cginc"
			#include "Common.cginc"

			//define vertex and fragment shader functions
			#pragma vertex vert
			#pragma fragment frag

			//tint of the texture
			fixed4 _Color;

			sampler2D _MainTex;
			uniform StructuredBuffer<Particle> particles;
			/*StructuredBuffer<int> indices;
			StructuredBuffer<float3> vertices;*/
			uniform StructuredBuffer<float3> quadPoints;
			uniform StructuredBuffer<float2> quadUVs;
			uniform float4x4 baseTransform;
			uniform float scale;
			uniform float simSize;

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				half4 col : COLOR;
				UNITY_FOG_COORDS(1)
				float4 pos : SV_POSITION;
			};

			v2f vert(uint id : SV_VertexID, uint instanceId : SV_InstanceID)
			{
				v2f o;
				Particle p = particles[instanceId];
				float3 v = quadPoints[id];
				o.uv = quadUVs[id];
				
				o.pos = mul(baseTransform,
					float4(p.pos * scale, 1)
				);
				float3 clippedCameraPos = float3(
					min(max(_WorldSpaceCameraPos.x, 0.0), simSize * scale),
					min(max(_WorldSpaceCameraPos.y, 0.0), simSize * scale),
					min(max(_WorldSpaceCameraPos.z, 0.0), simSize * scale)
				);
				float3 cameraDist = o.pos - clippedCameraPos;
				o.pos = mul(UNITY_MATRIX_P,
					mul(UNITY_MATRIX_V, o.pos) + float4(quadPoints[id] * float3(0.005 * scale, 0.005 * scale, 1), 0)
				);
				
				o.col = float4((float)(p.type % 2), 1, 1, 1);
				o.col.xyz /= (1.0 + (abs(cameraDist.x) + abs(cameraDist.y) + abs(cameraDist.z)) / simSize / scale);
				UNITY_TRANSFER_FOG(o, o.pos);
				return o;
			}

			fixed4 frag(v2f i) : SV_Target
			{
				fixed4 pc = tex2D(_MainTex, i.uv);
				fixed4 d = _Color * pc * i.col;
				clip(d.a - 0.01);
				return d;
			}

			////the vertex shader function
			//float4 vert(uint vertex_id: SV_VertexID, uint instance_id : SV_InstanceID) : SV_POSITION{
			//	//get vertex position
			//	int positionIndex = indices[vertex_id];
			//	float3 position = vertices[positionIndex];
			//	//add sphere position
			//	position += particles[instance_id];
			//	//convert the vertex position from world space to clip space
			//	return mul(UNITY_MATRIX_VP, float4(position, 1));
			//}

			////the fragment shader function
			//fixed4 frag() : SV_TARGET{
			//	//return the final color to be drawn on screen
			//	return _Color;
			//}

			ENDCG
		}
	}
	Fallback "VertexLit"
}
