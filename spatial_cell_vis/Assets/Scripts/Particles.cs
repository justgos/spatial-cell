using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Linq;
using UnityEngine;

public class Particles : MonoBehaviour
{
    public SimData simData;

    public GameObject model;
    private Mesh mesh;
    private float meshScale;
    public Material instancedMaterial;

    public Material material;

    ComputeBuffer quadPoints;
    ComputeBuffer quadUVs;

    uint[] drawArgs;
    ComputeBuffer argsBuffer;

    void Start()
    {
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

        // Argument buffer used by DrawMeshInstancedIndirect.
        drawArgs = new uint[5] { 0, 0, 0, 0, 0 };
        // Arguments for drawing mesh.
        // 0 == number of triangle indices, 1 == population, others are only relevant if drawing submeshes.
        mesh = model.GetComponent<MeshFilter>().sharedMesh;
        meshScale = model.transform.localScale.x;
        drawArgs[0] = (uint)mesh.GetIndexCount(0);
        drawArgs[1] = (uint)1;
        drawArgs[2] = (uint)mesh.GetIndexStart(0);
        drawArgs[3] = (uint)mesh.GetBaseVertex(0);
        argsBuffer = new ComputeBuffer(1, drawArgs.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        argsBuffer.SetData(drawArgs);
    }

    void Update()
    {
        if(simData.NumParticles != drawArgs[1])
        {
            drawArgs[1] = (uint)simData.NumParticles;
            argsBuffer.SetData(drawArgs);
        }
        instancedMaterial.SetMatrix("baseTransform", Matrix4x4.identity);
        instancedMaterial.SetFloat("scale", 10.0f);
        instancedMaterial.SetFloat("meshScale", meshScale);
        instancedMaterial.SetFloat("simSize", simData.SimSize);
        instancedMaterial.SetBuffer("particles", simData.frameBuffer);
        Graphics.DrawMeshInstancedIndirect(mesh, 0, instancedMaterial, new Bounds(Vector3.zero, Vector3.one * simData.SimSize * 10.0f), argsBuffer);
    }

    void OnRenderObject()
    {
        //material.SetMatrix("baseTransform", Matrix4x4.identity);
        //material.SetBuffer("quadPoints", quadPoints);
        //material.SetBuffer("quadUVs", quadUVs);
        //material.SetBuffer("particles", simData.frameBuffer);
        //material.SetFloat("scale", 10.0f);
        //material.SetFloat("simSize", simData.SimSize);
        //material.SetPass(0);
        //Graphics.DrawProceduralNow(MeshTopology.Triangles, 6, simData.frameBuffer.count);
    }

    void OnDestroy()
    {
        quadPoints.Dispose();
        quadUVs.Dispose();
        argsBuffer.Dispose();
    }
}
