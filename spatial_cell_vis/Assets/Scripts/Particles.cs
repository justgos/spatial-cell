using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Linq;
using UnityEngine;

public class Particles : MonoBehaviour
{
    public SimData simData;
    public FrameData frameData;

    public GameObject model;
    private Mesh mesh;
    private float meshScale;
    public Material instancedMaterial;
    public Material material;

    public GameObject debugVectorModel;
    private Mesh debugVectorMesh;
    private float debugVectorMeshScale;
    private float debugVectorMeshHeight;
    public Material debugVectorInstancedMaterial;

    private int targetParticleId = -1;
    public int TargetParticleId { set { targetParticleId = value; } }

    ComputeBuffer quadPoints;
    ComputeBuffer quadUVs;

    uint[] drawArgs;
    ComputeBuffer argsBuffer;

    uint[] debugVectorDrawArgs;
    ComputeBuffer debugVectorArgsBuffer;

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

        mesh = model.GetComponent<MeshFilter>().sharedMesh;
        meshScale = model.transform.localScale.x;
        // Argument buffer used by DrawMeshInstancedIndirect.
        // Arguments for drawing mesh.
        // 0 == number of triangle indices, 1 == population, others are only relevant if drawing submeshes.
        drawArgs = new uint[5] { 0, 0, 0, 0, 0 };
        drawArgs[0] = (uint)mesh.GetIndexCount(0);
        drawArgs[1] = (uint)1;
        drawArgs[2] = (uint)mesh.GetIndexStart(0);
        drawArgs[3] = (uint)mesh.GetBaseVertex(0);
        argsBuffer = new ComputeBuffer(1, drawArgs.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        argsBuffer.SetData(drawArgs);

        debugVectorMesh = debugVectorModel.GetComponent<MeshFilter>().sharedMesh;
        debugVectorMeshScale = debugVectorModel.transform.localScale.y;
        debugVectorMeshHeight = (debugVectorMesh.bounds.center.y + debugVectorMesh.bounds.extents.y) * debugVectorMeshScale;

        debugVectorDrawArgs = new uint[5] { 0, 0, 0, 0, 0 };
        debugVectorDrawArgs[0] = (uint)debugVectorMesh.GetIndexCount(0);
        debugVectorDrawArgs[1] = (uint)1;
        debugVectorDrawArgs[2] = (uint)debugVectorMesh.GetIndexStart(0);
        debugVectorDrawArgs[3] = (uint)debugVectorMesh.GetBaseVertex(0);
        debugVectorArgsBuffer = new ComputeBuffer(1, debugVectorDrawArgs.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        debugVectorArgsBuffer.SetData(debugVectorDrawArgs);
    }

    void Update()
    {
        if(frameData.NumParticles != drawArgs[1])
        {
            drawArgs[1] = (uint)frameData.NumParticles;
            argsBuffer.SetData(drawArgs);
            debugVectorDrawArgs[1] = (uint)frameData.NumParticles;
            debugVectorArgsBuffer.SetData(debugVectorDrawArgs);
        }
        instancedMaterial.SetMatrix("baseTransform", Matrix4x4.identity);
        instancedMaterial.SetFloat("scale", 10.0f);
        instancedMaterial.SetFloat("meshScale", meshScale);
        instancedMaterial.SetFloat("simSize", simData.SimSize);
        instancedMaterial.SetInt("targetParticleId", targetParticleId);
        instancedMaterial.SetBuffer("particles", frameData.ParticleBuffer);
        Graphics.DrawMeshInstancedIndirect(mesh, 0, instancedMaterial, new Bounds(Vector3.one * simData.SimSize * 10.0f * 0.5f, Vector3.one * simData.SimSize * 10.0f), argsBuffer);

        debugVectorInstancedMaterial.SetMatrix("baseTransform", Matrix4x4.identity);
        debugVectorInstancedMaterial.SetFloat("scale", 10.0f);
        debugVectorInstancedMaterial.SetFloat("meshScale", debugVectorMeshScale);
        debugVectorInstancedMaterial.SetFloat("meshHeight", debugVectorMeshHeight);
        debugVectorInstancedMaterial.SetFloat("simSize", simData.SimSize);
        debugVectorInstancedMaterial.SetBuffer("particles", frameData.ParticleBuffer);
        Graphics.DrawMeshInstancedIndirect(debugVectorMesh, 0, debugVectorInstancedMaterial, new Bounds(Vector3.one * simData.SimSize * 10.0f * 0.5f, Vector3.one * simData.SimSize * 10.0f), debugVectorArgsBuffer);
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
        debugVectorArgsBuffer.Dispose();
    }
}
