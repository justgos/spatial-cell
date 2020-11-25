using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.UI;

public class SimData : MonoBehaviour
{
    public GameObject spherePrefab;

    [StructLayout(LayoutKind.Sequential)]
    public struct Vector3_
    {
        //[MarshalAs(UnmanagedType.fl)]
        public float x;
        public float y;
        public float z;
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct Vector4_
    {
        public float x;
        public float y;
        public float z;
        public float w;
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct ParticleInteraction
    {
        public int type;
        public int partnerId;
    };

    [StructLayout(LayoutKind.Explicit, Size = 112)]
    public struct Particle
    {
        [FieldOffset(0)]  public int id;
        [FieldOffset(4)]  public int type;
        [FieldOffset(8)]  public int flags;
        [FieldOffset(12)] public Vector3_ pos;
        [FieldOffset(32)] public Vector4_ rot;
        [FieldOffset(48)] public Vector3_ velocity;
        [FieldOffset(60)] public int nActiveInteractions;
        [FieldOffset(64)] public ParticleInteraction interaction1;
        [FieldOffset(72)] public ParticleInteraction interaction2;
        [FieldOffset(80)] public ParticleInteraction interaction3;
        [FieldOffset(88)] public ParticleInteraction interaction4;
        [FieldOffset(96)] public Vector4_ debugVector;
    };

    public class SimFrame
    {
        public uint numParticles;
        //public Particle[] particles;
        public NativeArray<Particle> particles;
    }

    private float simSize;
    public float SimSize { get { return simSize; } }
    private int particleBufferSize;
    private List<SimFrame> frames = new List<SimFrame>();
    private int numParticles;
    public int NumParticles { get { return numParticles; } }
    public ComputeBuffer frameBuffer;

    public Slider frameSlider;
    public Text frameNumberText;

    private float lastFrameTime = 0;
    private bool playing = false;
    private float playbackFps = 30;

    void Start()
    {
        System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
        sw.Start();
        using (var fs = File.OpenRead(@"../spatial_cell_sim/results/frames.dat"))
        {
            var br = new BinaryReader(fs);
            simSize = br.ReadSingle();
            particleBufferSize = br.ReadInt32();
            var particleStructSize = Marshal.SizeOf(new Particle());
            Debug.Log("particleStructSize " +  particleStructSize);
            frameBuffer = new ComputeBuffer(particleBufferSize, particleStructSize);
            while (br.BaseStream.Position != br.BaseStream.Length)
            //for (var j = 0; j < nFrames; j++)
            {
                var frame = new SimFrame();
                frame.numParticles = br.ReadUInt32();
                //Debug.Log("frame.numParticles " + frame.numParticles);
                //frame.particles = new Particle[frame.numParticles];
                frame.particles = new NativeArray<Particle>((int)frame.numParticles, Allocator.Persistent);
                var bytes = br.ReadBytes((int)(particleStructSize * frame.numParticles));
                //Marshal.Copy(, 0, (IntPtr)frame.particles, 0, (int)(particleStructSize * frame.numParticles));

                //Debug.Log("flags " + bytes[8] + bytes[9] + bytes[10] + bytes[11]);

                unsafe
                {
                    fixed (void* bytesPointer = bytes)
                    {
                        //UnsafeUtility.CopyStructureToPtr((byte*)bytes[0], frame.particles.GetUnsafePtr());
                        UnsafeUtility.MemCpy(frame.particles.GetUnsafePtr(), bytesPointer, UnsafeUtility.SizeOf<Particle>() * frame.numParticles);
                    }
                }
                //if(frames.Count == 0)
                //{
                //    Debug.Log("id " + frame.particles[0].id + ", " + bytes[0]);
                //    Debug.Log("type " + frame.particles[0].type + ", " + bytes[4]);
                //    Debug.Log("flags " + frame.particles[0].flags + ", " + bytes[8] + bytes[9] + bytes[10] + bytes[11]);
                //    Debug.Log("pos.x " + frame.particles[0].pos.x);
                //}

                //UnsafeUtility.
                //NativeArray.
                //for (var i = 0; i < frame.numParticles; i++)
                //    frame.particles[i] = Marshal.PtrToStructure<Particle>(Marshal.UnsafeAddrOfPinnedArrayElement(bytes, particleStructSize * i));

                frames.Add(frame);
            }
            frameSlider.maxValue = frames.Count - 1;
            frameNumberText.text = frameSlider.value.ToString();
            frameBuffer.SetData(frames[0].particles);
            numParticles = frames[0].particles.Length;
        }
        sw.Stop();
        Debug.Log("Frames loaded in " + ((double)sw.ElapsedTicks / System.Diagnostics.Stopwatch.Frequency) + "s");
    }

    void Update()
    {
        if(Input.GetKeyDown(KeyCode.Space))
        {
            this.playing = !playing;
            this.lastFrameTime = Time.time;
        }
        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            SetFrame(frameSlider.value - (Input.GetKey(KeyCode.LeftShift) ? 5 : 1));
        }
        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            SetFrame(frameSlider.value + (Input.GetKey(KeyCode.LeftShift) ? 5 : 1));
        }

        if (this.playing)
        {
            if(this.lastFrameTime + 1.0f / this.playbackFps <= Time.time)
            {
                this.lastFrameTime = Time.time;
                SetFrame(frameSlider.value + 1);
            }
        }
    }

    void SetFrame(float frameNum)
    {
        if (frameNum < 0)
            frameNum = this.frames.Count + frameNum;
        else if (frameNum >= this.frames.Count)
            frameNum = frameNum % this.frames.Count;
        frameSlider.value = frameNum;
    }

    public void ChangeFrame(float frameNum)
    {
        frameBuffer.SetData(frames[(int)frameNum].particles);
        numParticles = frames[(int)frameNum].particles.Length;
        frameNumberText.text = frameSlider.value.ToString();
    }

    //void OnDrawGizmosSelected()
    //{
    //    if (frames.Count < 1)
    //        return;
    //    // Draw a yellow sphere at the transform's position
    //    for (var j = 0; j < 3; j++)
    //    {
    //        var frame = frames[j];
    //        if(j == 0)
    //            Gizmos.color = Color.red;
    //        else if (j == 1)
    //            Gizmos.color = Color.green;
    //        else if (j == 2)
    //            Gizmos.color = Color.blue;
    //        for (var i = 0; i < frame.positions.Length; i++)
    //        {
    //            Gizmos.DrawSphere(frame.positions[i], 0.05f);
    //        }
    //    }
    //}

    void OnDestroy()
    {
        frameBuffer.Dispose();
        frames.ForEach(f => f.particles.Dispose());
    }
}
