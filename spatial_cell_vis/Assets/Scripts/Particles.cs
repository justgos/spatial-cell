using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Linq;
using UnityEngine;

public class Particles : MonoBehaviour
{
    public SimData simData;

    public Mesh mesh;
    public Material material;

    //private ComputeBuffer meshVertices;
    //private ComputeBuffer meshIndices;
    ComputeBuffer quadPoints;
    ComputeBuffer quadUVs;

    void Start()
    {
        //meshIndices = new ComputeBuffer(mesh.triangles.Length, sizeof(int));
        //meshIndices.SetData(mesh.triangles);
        //material.SetBuffer("indices", meshIndices);
        //const float meshScale = 0.05f;
        //Vector3[] positions = mesh.vertices.Select(p => p * meshScale).ToArray();
        //meshVertices = new ComputeBuffer(positions.Length, sizeof(float) * 3);
        //meshVertices.SetData(positions);
        //material.SetBuffer("vertices", meshVertices);
        quadPoints = new ComputeBuffer(6, Marshal.SizeOf(new Vector3()));
        quadPoints.SetData(new[] {
                new Vector3(-0.5f, 0.5f),
                new Vector3(0.5f, 0.5f),
                new Vector3(0.5f, -0.5f),
                new Vector3(0.5f, -0.5f),
                new Vector3(-0.5f, -0.5f),
                new Vector3(-0.5f, 0.5f),
            });

        quadUVs = new ComputeBuffer(6, Marshal.SizeOf(new Vector2()));
        quadUVs.SetData(new[] {
                new Vector2(0.0f, 1.0f),
                new Vector2(1.0f, 1.0f),
                new Vector2(1.0f, 0.0f),
                new Vector2(1.0f, 0.0f),
                new Vector2(0.0f, 0.0f),
                new Vector2(0.0f, 1.0f),
            });
    }

    void OnRenderObject()
    {
        //material.SetBuffer("indices", meshIndices);
        //material.SetBuffer("vertices", meshVertices);
        material.SetMatrix("baseTransform", Matrix4x4.identity);
        material.SetBuffer("quadPoints", quadPoints);
        material.SetBuffer("quadUVs", quadUVs);
        material.SetBuffer("particles", simData.frameBuffer);
        material.SetFloat("scale", 10.0f);
        material.SetFloat("simSize", simData.SimSize);
        material.SetPass(0);
        Graphics.DrawProceduralNow(MeshTopology.Triangles, 6, simData.frameBuffer.count);
        //Graphics.DrawProcedural(material, new Bounds(Vector3.zero, Vector3.one * 2), MeshTopology.Triangles, mesh.vertexCount, simData.frameBuffer.count / 3);
    }

    void OnDestroy()
    {
        //meshIndices.Dispose();
        //meshVertices.Dispose();
        quadPoints.Dispose();
        quadUVs.Dispose();
    }
}
