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
    private SimData.SimFrame frame = null;
    public SimData.SimFrame Frame { get { return frame; } }


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

    public void SetData<T> (NativeArray<T> items, int count, SimData.SimFrame simFrame) where T : struct
    {
        numParticles = count;
        particleBuffer.SetData(items);
        frame = simFrame;
    }

    void OnDestroy()
    {
        particleBuffer?.Dispose();
    }
}
