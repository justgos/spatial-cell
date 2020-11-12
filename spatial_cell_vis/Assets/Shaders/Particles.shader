Shader "Custom/Particles"
{
	Properties{
		_MainTex("Texture", 2D) = "white" {}
		[HDR] _Color("Tint", Color) = (0, 0, 0, 1)
	}

	SubShader{
		//the material is completely non-transparent and is rendered at the same time as the other opaque geometry
		Tags{ "RenderType" = "Opaque" "Queue" = "Geometry" }
		//Tags{ "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" }

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

			//define vertex and fragment shader functions
			#pragma vertex vert
			#pragma fragment frag

			//tint of the texture
			fixed4 _Color;

			sampler2D _MainTex;
			uniform StructuredBuffer<float3> particles;
			/*StructuredBuffer<int> indices;
			StructuredBuffer<float3> vertices;*/
			uniform StructuredBuffer<float3> quadPoints;
			uniform StructuredBuffer<float2> quadUVs;
			float4x4 baseTransform;

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
				float3 p = particles[instanceId];
				float3 v = quadPoints[id];
				o.uv = quadUVs[id];
				// Discard near-zero cells and the topmost layer of simulation
				/*if (p.val < cutoffValue || p.position.y > 150) {
					o.pos = float4(0, 0, 0, 0);
				}
				else {*/
					o.pos = mul(UNITY_MATRIX_P,
						mul(UNITY_MATRIX_V,
							mul(baseTransform,
								float4(p, 1)
							)) + float4(quadPoints[id] * float3(0.005, 0.005, 1), 0));
				//}
				o.col = float4(1, 1, 1, 1);
				UNITY_TRANSFER_FOG(o, o.pos);
				return o;
			}

			fixed4 frag(v2f i) : SV_Target
			{
				fixed4 pc = tex2D(_MainTex, i.uv);
				fixed4 d = _Color * pc * i.col;
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
