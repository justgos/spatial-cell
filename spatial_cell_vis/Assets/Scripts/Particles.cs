using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Linq;
using UnityEngine;
using UnityEngine.UI.Extensions;

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

    public RangeSlider particleVisibleRangeXSlider;
    public RangeSlider particleVisibleRangeYSlider;
    public RangeSlider particleVisibleRangeZSlider;

    private int particleTypeFilter = 0x0000ffff;  // 0x7fffffff;

    private int targetParticleId = -1;
    public int TargetParticleId { set { targetParticleId = value; } }

    public bool drawInstanced = true;

    ComputeBuffer quadPoints;
    ComputeBuffer quadUVs;

    uint[] drawArgs;
    ComputeBuffer argsBuffer;

    uint[] debugVectorDrawArgs;
    ComputeBuffer debugVectorArgsBuffer;

    void Start()
    {
        instancedMaterial = Instantiate(instancedMaterial);
        debugVectorInstancedMaterial = Instantiate(debugVectorInstancedMaterial);
        material = Instantiate(material);

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

        SetModel(model);

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

    public void SetModel(GameObject model)
    {
        this.model = model;
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
    }

    public void ToggleParticleType(int type)
    {
        particleTypeFilter = particleTypeFilter ^ (1 << type);
        Debug.Log(string.Format("{0}", Convert.ToString(particleTypeFilter, 2)));
    }

    void Update()
    {
        if (frameData.Frame == null)
            return;

        if (frameData.NumParticles != drawArgs[1])
        {
            drawArgs[1] = (uint)frameData.NumParticles;
            argsBuffer.SetData(drawArgs);
            debugVectorDrawArgs[1] = (uint)frameData.NumParticles;
            debugVectorArgsBuffer.SetData(debugVectorDrawArgs);
        }

        if (drawInstanced)
        {
            instancedMaterial.SetMatrix("baseTransform", Matrix4x4.identity);
            instancedMaterial.SetFloat("scale", 10.0f);
            instancedMaterial.SetFloat("meshScale", meshScale);
            instancedMaterial.SetFloat("simSize", simData.SimSize);
            instancedMaterial.SetFloat("visibleMinX", particleVisibleRangeXSlider.LowValue);
            instancedMaterial.SetFloat("visibleMaxX", particleVisibleRangeXSlider.HighValue);
            instancedMaterial.SetFloat("visibleMinY", particleVisibleRangeYSlider.LowValue);
            instancedMaterial.SetFloat("visibleMaxY", particleVisibleRangeYSlider.HighValue);
            instancedMaterial.SetFloat("visibleMinZ", particleVisibleRangeZSlider.LowValue);
            instancedMaterial.SetFloat("visibleMaxZ", particleVisibleRangeZSlider.HighValue);
            instancedMaterial.SetInt("particleTypeFilter", particleTypeFilter);
            instancedMaterial.SetInt("targetParticleId", targetParticleId);
            instancedMaterial.SetBuffer("particles", frameData.ParticleBuffer);
            Graphics.DrawMeshInstancedIndirect(mesh, 0, instancedMaterial, new Bounds(Vector3.one * simData.SimSize * 10.0f * 0.5f, Vector3.one * simData.SimSize * 10.0f), argsBuffer);

            debugVectorInstancedMaterial.SetMatrix("baseTransform", Matrix4x4.identity);
            debugVectorInstancedMaterial.SetFloat("scale", 10.0f);
            debugVectorInstancedMaterial.SetFloat("meshScale", debugVectorMeshScale);
            debugVectorInstancedMaterial.SetFloat("meshHeight", debugVectorMeshHeight);
            debugVectorInstancedMaterial.SetFloat("simSize", simData.SimSize);
            debugVectorInstancedMaterial.SetFloat("visibleMinX", particleVisibleRangeXSlider.LowValue);
            debugVectorInstancedMaterial.SetFloat("visibleMaxX", particleVisibleRangeXSlider.HighValue);
            debugVectorInstancedMaterial.SetFloat("visibleMinY", particleVisibleRangeYSlider.LowValue);
            debugVectorInstancedMaterial.SetFloat("visibleMaxY", particleVisibleRangeYSlider.HighValue);
            debugVectorInstancedMaterial.SetFloat("visibleMinZ", particleVisibleRangeZSlider.LowValue);
            debugVectorInstancedMaterial.SetFloat("visibleMaxZ", particleVisibleRangeZSlider.HighValue);
            debugVectorInstancedMaterial.SetInt("particleTypeFilter", particleTypeFilter);
            debugVectorInstancedMaterial.SetBuffer("particles", frameData.ParticleBuffer);
            Graphics.DrawMeshInstancedIndirect(debugVectorMesh, 0, debugVectorInstancedMaterial, new Bounds(Vector3.one * simData.SimSize * 10.0f * 0.5f, Vector3.one * simData.SimSize * 10.0f), debugVectorArgsBuffer);
        }

        // FIXME: `argsBuffer` is not read out immediately, and so the setting above gets rewritten
        // and? ruins the previous render
        foreach (var entry in frameData.filteredBuffers)
        {
            drawArgs[1] = (uint)frameData.filteredBuffer_Sizes[entry.Key];
            argsBuffer.SetData(drawArgs);

            instancedMaterial.SetMatrix("baseTransform", Matrix4x4.identity);
            instancedMaterial.SetFloat("scale", 10.0f);
            instancedMaterial.SetFloat("meshScale", meshScale);
            instancedMaterial.SetFloat("simSize", simData.SimSize);
            instancedMaterial.SetFloat("visibleMinX", particleVisibleRangeXSlider.LowValue);
            instancedMaterial.SetFloat("visibleMaxX", particleVisibleRangeXSlider.HighValue);
            instancedMaterial.SetFloat("visibleMinY", particleVisibleRangeYSlider.LowValue);
            instancedMaterial.SetFloat("visibleMaxY", particleVisibleRangeYSlider.HighValue);
            instancedMaterial.SetFloat("visibleMinZ", particleVisibleRangeZSlider.LowValue);
            instancedMaterial.SetFloat("visibleMaxZ", particleVisibleRangeZSlider.HighValue);
            instancedMaterial.SetInt("particleTypeFilter", particleTypeFilter);
            instancedMaterial.SetInt("targetParticleId", targetParticleId);
            instancedMaterial.SetBuffer("particles", entry.Value);
            Graphics.DrawMeshInstancedIndirect(mesh, 0, instancedMaterial, new Bounds(Vector3.one * simData.SimSize * 10.0f * 0.5f, Vector3.one * simData.SimSize * 10.0f), argsBuffer);
        }

        //debugVectorInstancedMaterial.SetMatrix("baseTransform", Matrix4x4.identity);
        //debugVectorInstancedMaterial.SetFloat("scale", 10.0f);
        //debugVectorInstancedMaterial.SetFloat("meshScale", debugVectorMeshScale);
        //debugVectorInstancedMaterial.SetFloat("meshHeight", debugVectorMeshHeight);
        //debugVectorInstancedMaterial.SetFloat("simSize", simData.SimSize);
        //debugVectorInstancedMaterial.SetFloat("visibleMinX", particleVisibleRangeXSlider.LowValue);
        //debugVectorInstancedMaterial.SetFloat("visibleMaxX", particleVisibleRangeXSlider.HighValue);
        //debugVectorInstancedMaterial.SetFloat("visibleMinY", particleVisibleRangeYSlider.LowValue);
        //debugVectorInstancedMaterial.SetFloat("visibleMaxY", particleVisibleRangeYSlider.HighValue);
        //debugVectorInstancedMaterial.SetFloat("visibleMinZ", particleVisibleRangeZSlider.LowValue);
        //debugVectorInstancedMaterial.SetFloat("visibleMaxZ", particleVisibleRangeZSlider.HighValue);
        //debugVectorInstancedMaterial.SetInt("particleTypeFilter", particleTypeFilter);
        //debugVectorInstancedMaterial.SetBuffer("particles", frameData.ParticleBuffer);
        //Graphics.DrawMeshInstancedIndirect(debugVectorMesh, 0, debugVectorInstancedMaterial, new Bounds(Vector3.one * simData.SimSize * 10.0f * 0.5f, Vector3.one * simData.SimSize * 10.0f), debugVectorArgsBuffer);
    }

    void OnRenderObject()
    {
        if (!drawInstanced)
        {
            material.SetMatrix("baseTransform", Matrix4x4.identity);
            material.SetBuffer("quadPoints", quadPoints);
            material.SetBuffer("quadUVs", quadUVs);
            material.SetBuffer("particles", frameData.ParticleBuffer);
            material.SetFloat("scale", 10.0f);
            material.SetFloat("simSize", simData.SimSize);
            material.SetFloat("visibleMinX", particleVisibleRangeXSlider.LowValue);
            material.SetFloat("visibleMaxX", particleVisibleRangeXSlider.HighValue);
            material.SetFloat("visibleMinY", particleVisibleRangeYSlider.LowValue);
            material.SetFloat("visibleMaxY", particleVisibleRangeYSlider.HighValue);
            material.SetFloat("visibleMinZ", particleVisibleRangeZSlider.LowValue);
            material.SetFloat("visibleMaxZ", particleVisibleRangeZSlider.HighValue);
            material.SetInt("particleTypeFilter", particleTypeFilter);
            //Debug.Log(string.Format("{0}", Convert.ToString(particleTypeFilter, 2)));
            material.SetInt("targetParticleId", targetParticleId);
            material.SetBuffer("particles", frameData.ParticleBuffer);
            material.SetPass(0);
            Graphics.DrawProceduralNow(MeshTopology.Triangles, 6, frameData.NumParticles);
        }
    }

    void OnDestroy()
    {
        if (quadPoints != null)
        {
            quadPoints.Dispose();
            quadUVs.Dispose();
            argsBuffer.Dispose();
            debugVectorArgsBuffer.Dispose();
        }
    }
}
