﻿using System;
using System.Collections;
using System.Collections.Generic;
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


    void Start()
    {
        
    }

    public void Init(int count, int itemSize)
    {
        particleBuffer = new ComputeBuffer(count, itemSize);
    }

    void Update()
    {
        
    }

    public void SetData<T> (Nullable<NativeArray<T>> items, int count, Nullable<SimData.SimFrame> simFrame) where T : struct
    {
        numParticles = count;
        if(items.HasValue)
            particleBuffer.SetData(items.Value);
        frame = simFrame;
    }

    void OnDestroy()
    {
        particleBuffer?.Dispose();
    }
}
