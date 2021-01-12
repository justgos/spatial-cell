using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

public class FrameData : MonoBehaviour
{
    private int numParticles = 0;
    public int NumParticles { get { return numParticles; } }
    private ComputeBuffer particleBuffer = null;
    public ComputeBuffer ParticleBuffer { get { return particleBuffer; } }
    private Nullable<SimData.SimFrame> frame = null;
    public Nullable<SimData.SimFrame> Frame { get { return frame; } }

    public ComputeShader filterParticles;
    public Dictionary<int, ComputeBuffer> filteredBuffers = new Dictionary<int, ComputeBuffer>();
    public Dictionary<int, ComputeBuffer> filteredBufferArgs = new Dictionary<int, ComputeBuffer>();

    void Start()
    {
        
    }

    //public void Init(int count, int itemSize)
    //{
    //    particleBuffer = new ComputeBuffer(count, itemSize);
    //}

    void Update()
    {
        
    }

    public void SetData<T> (Nullable<NativeArray<T>> items, int count, Nullable<SimData.SimFrame> simFrame) where T : struct
    {
        numParticles = count;
        if (items.HasValue)
        {
            int bufferSize = particleBuffer != null ? particleBuffer.count : 1;
            while (bufferSize < items.Value.Length)
                bufferSize *= 2;
            // Grow the bufeer if necessary
            if(bufferSize != particleBuffer?.count)
            {
                particleBuffer?.Dispose();
                particleBuffer = new ComputeBuffer(count, Marshal.SizeOf(typeof(T)));
            }
            particleBuffer.SetData(items.Value);

            if (filterParticles != null)
            {
                System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                sw.Start();

                // TODO: dynamically grow/clear instead of re-creating
                StartCoroutine(ClearBuffers(filteredBuffers, filteredBufferArgs));
                filteredBuffers = new Dictionary<int, ComputeBuffer>();
                filteredBufferArgs = new Dictionary<int, ComputeBuffer>();

                var typeId = 110;
                var buffer = new ComputeBuffer(bufferSize, Marshal.SizeOf(typeof(T)), ComputeBufferType.Append);
                buffer.SetCounterValue(0);
                var argBuffer = new ComputeBuffer(5, sizeof(int), ComputeBufferType.IndirectArguments);
                filterParticles.SetBuffer(0, "particles", particleBuffer);
                filterParticles.SetBuffer(0, "filteredParticles", buffer);
                filterParticles.SetInt("type", typeId);
                filterParticles.SetInt("count", items.Value.Length);
                filterParticles.Dispatch(0, bufferSize / 8, 1, 1);
                int[] args = new int[] { 0, 1, 0, 0, 0 };
                argBuffer.SetData(args);
                ComputeBuffer.CopyCount(buffer, argBuffer, 0);
                argBuffer.GetData(args);
                Debug.Log(string.Format("Filtered {0} particles of type {1}", args[0], typeId));
                args[1] = args[0];
                argBuffer.SetData(args);
                filteredBuffers.Add(typeId, buffer);
                filteredBufferArgs.Add(typeId, argBuffer);

                sw.Stop();
                Debug.Log("Filtered particles in " + ((double)sw.ElapsedTicks / System.Diagnostics.Stopwatch.Frequency) + "s");
            }
        }
        frame = simFrame;
    }

    IEnumerator ClearBuffers(Dictionary<int, ComputeBuffer> filteredBuffers, Dictionary<int, ComputeBuffer> filteredBufferArgs)
    {
        yield return new WaitForEndOfFrame();

        foreach (var entry in filteredBuffers)
            entry.Value.Dispose();
        foreach (var entry in filteredBufferArgs)
            entry.Value.Dispose();
    }

    void OnDestroy()
    {
        particleBuffer?.Dispose();
        particleBuffer = null;

        foreach (var entry in filteredBuffers)
            entry.Value.Dispose();
        foreach (var entry in filteredBufferArgs)
            entry.Value.Dispose();
    }
}
