using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.UI;
using UnityEngine.UI.Extensions;


public class SimData : MonoBehaviour
{
    public static readonly int PARTICLE_FLAG_ACTIVE = 0x0001;

    public GameObject spherePrefab;

    [StructLayout(LayoutKind.Sequential)]
    public struct Vector3_
    {
        //[MarshalAs(UnmanagedType.fl)]
        public float x;
        public float y;
        public float z;

        public Vector3 UnityVector()
        {
            return new Vector3(x, y, z);
        }
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct Vector4_
    {
        public float x;
        public float y;
        public float z;
        public float w;

        public Quaternion UnityQuaternion()
        {
            return new Quaternion(x, y, z, w);
        }
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct ParticleInteraction
    {
        public int type;
        public int partnerId;
    };

    [StructLayout(LayoutKind.Explicit, Size = 144)]
    unsafe public struct Particle
    {
        [FieldOffset(0)]  public int id;
        [FieldOffset(4)]  public int type;
        [FieldOffset(8)]  public int flags;
        [FieldOffset(12)] public Vector3_ pos;
        [FieldOffset(24)] public fixed float __padding1[2];
        [FieldOffset(32)] public Vector4_ rot;
        [FieldOffset(48)] public Vector3_ velocity;
        [FieldOffset(64)] public Vector4_ angularVelocity;
        [FieldOffset(80)] public int nActiveInteractions;
        [FieldOffset(84)] public ParticleInteraction interaction1;
        [FieldOffset(92)] public ParticleInteraction interaction2;
        [FieldOffset(100)] public ParticleInteraction interaction3;
        [FieldOffset(108)] public ParticleInteraction interaction4;
        [FieldOffset(128)] public Vector4_ debugVector;
    };

    [StructLayout(LayoutKind.Explicit, Size = 352)]
    unsafe public struct MetabolicParticle
    {
        [FieldOffset(0)] public int id;
        [FieldOffset(4)] public int type;
        [FieldOffset(8)] public int flags;
        [FieldOffset(12)] public Vector3_ pos;
        [FieldOffset(24)] public fixed float __padding1[2];
        [FieldOffset(32)] public Vector4_ rot;
        [FieldOffset(48)] public Vector3_ velocity;
        [FieldOffset(64)] public Vector4_ angularVelocity;
        [FieldOffset(80)] public int nActiveInteractions;
        [FieldOffset(84)] public ParticleInteraction interaction1;
        [FieldOffset(92)] public ParticleInteraction interaction2;
        [FieldOffset(100)] public ParticleInteraction interaction3;
        [FieldOffset(108)] public ParticleInteraction interaction4;
        [FieldOffset(128)] public Vector4_ debugVector;
        [FieldOffset(144)] public fixed float metabolites[50];
    };

    public class SimFrame
    {
        public uint numParticles;
        //public Particle[] particles;
        public NativeArray<Particle> particles;
        public uint numMetabolicParticles;
        public NativeArray<MetabolicParticle> metabolicParticles;
    }

    private float simSize;
    public float SimSize { get { return simSize; } }
    private List<SimFrame> frames = new List<SimFrame>();

    public FrameData particleFrameData;
    public FrameData metabolicParticleFrameData;

    public Slider frameSlider;
    public Text frameNumberText;

    private float lastFrameTime = 0;
    private bool playing = false;
    private float playbackFps = 30;

    public RangeSlider particleVisibleRangeXSlider;
    public RangeSlider particleVisibleRangeYSlider;
    public RangeSlider particleVisibleRangeZSlider;

    void Start()
    {
        System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
        sw.Start();


        //using (var mmf = MemoryMappedFile.OpenExisting("spatial_cell_buf"))
        //{
        //    mmf.CreateViewStream();
        //    using (var accessor = mmf.CreateViewAccessor(0, 2 * 1024 * 1024 * 1023))
        //    {
        //        int colorSize = Marshal.SizeOf(typeof(MyColor));
        //        MyColor color;

        //        // Make changes to the view.
        //        for (long i = 0; i < 1500000; i += colorSize)
        //        {
        //            accessor.Read(i, out color);
        //            color.Brighten(20);
        //            accessor.Write(i, ref color);
        //        }
        //    }
        //}

        using (var fs = File.OpenRead(@"../spatial_cell_sim/results/frames.dat"))
        //using (var mmf = MemoryMappedFile.OpenExisting("spatial_cell_buf"))
        {
            //using (var accessor = mmf.CreateViewAccessor(0, 2 * 1024 * 1024 * 1023))
            //{
                //var br = new BinaryReader(fs);
                //var memStream = mmf.CreateViewStream();
                var br = new BinaryReader(fs);
                simSize = br.ReadSingle();
                particleVisibleRangeXSlider.MaxValue = simSize;
                particleVisibleRangeYSlider.MaxValue = simSize;
                particleVisibleRangeZSlider.MaxValue = simSize;

                int particleBufferSize = br.ReadInt32();
                var particleStructSize = Marshal.SizeOf(new Particle());
                Debug.Log("numParticles " + particleBufferSize);
                Debug.Log("particleStructSize " + particleStructSize);
                particleFrameData.Init(particleBufferSize, particleStructSize);
                int metabolicParticleBufferSize = br.ReadInt32();
                var metabolicParticleStructSize = Marshal.SizeOf(new MetabolicParticle());
                Debug.Log("numMetabolicParticles " + metabolicParticleBufferSize);
                Debug.Log("metabolicParticleStructSize " + metabolicParticleStructSize);
                metabolicParticleFrameData.Init(metabolicParticleBufferSize, metabolicParticleStructSize);
                while (br.BaseStream.Position != br.BaseStream.Length)
                //for (var j = 0; j < nFrames; j++)
                {
                    var frame = new SimFrame();
                    frame.numParticles = br.ReadUInt32();
                    if (frame.numParticles < 1)
                        break;
                    //Debug.Log("frame.numParticles " + frame.numParticles);
                    //frame.particles = new Particle[frame.numParticles];
                    frame.particles = new NativeArray<Particle>((int)frame.numParticles, Allocator.Persistent);
                    int frameSize = (int)(particleStructSize * frame.numParticles);
                    var bytes = br.ReadBytes(frameSize);
                    //Marshal.Copy(, 0, (IntPtr)frame.particles, 0, (int)(particleStructSize * frame.numParticles));

                    //Debug.Log("flags " + bytes[8] + bytes[9] + bytes[10] + bytes[11]);

                    unsafe
                    {
                        //byte* ptr = (byte*)0;
                        //accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
                        fixed (void* bytesPointer = bytes)
                        {
                            //UnsafeUtility.CopyStructureToPtr((byte*)bytes[0], frame.particles.GetUnsafePtr());
                            UnsafeUtility.MemCpy(frame.particles.GetUnsafePtr(), bytesPointer, UnsafeUtility.SizeOf<Particle>() * frame.numParticles);
                            //UnsafeUtility.MemCpy(frame.particles.GetUnsafePtr(), ptr + br.BaseStream.Position, UnsafeUtility.SizeOf<Particle>() * frame.numParticles);
                        }
                        //accessor.SafeMemoryMappedViewHandle.ReleasePointer();
                    }
                    //br.BaseStream.Position += frameSize;

                    frame.numMetabolicParticles = br.ReadUInt32();
                    frame.metabolicParticles = new NativeArray<MetabolicParticle>((int)frame.numMetabolicParticles, Allocator.Persistent);
                    frameSize = (int)(metabolicParticleStructSize * frame.numMetabolicParticles);
                    bytes = br.ReadBytes(frameSize);

                    unsafe
                    {
                        fixed (void* bytesPointer = bytes)
                        {
                            UnsafeUtility.MemCpy(frame.metabolicParticles.GetUnsafePtr(), bytesPointer, UnsafeUtility.SizeOf<MetabolicParticle>() * frame.numMetabolicParticles);
                        }
                    }

                    //UnsafeUtility.
                    //NativeArray.
                    //for (var i = 0; i < frame.numParticles; i++)
                    //    frame.particles[i] = Marshal.PtrToStructure<Particle>(Marshal.UnsafeAddrOfPinnedArrayElement(bytes, particleStructSize * i));

                    frames.Add(frame);

                    //unsafe
                    //{
                    //    fixed (float* m = frame.metabolicParticles.ToArray()[0].metabolites)
                    //    {
                    //        Debug.Log("frame.metabolicParticles " + m[0] + ", " + m[1] + ", " + m[2] + ", " + m[3]);
                    //    }
                    //}
                    //return;
                }
                frameSlider.maxValue = frames.Count - 1;
                frameNumberText.text = frameSlider.value.ToString();
                particleFrameData.SetData(frames[0].particles, frames[0].particles.Length, frames[0]);
                metabolicParticleFrameData.SetData(frames[0].metabolicParticles, frames[0].metabolicParticles.Length, frames[0]);
            //}
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
        particleFrameData.SetData(frames[(int)frameNum].particles, frames[(int)frameNum].particles.Length, frames[(int)frameNum]);
        metabolicParticleFrameData.SetData(frames[(int)frameNum].metabolicParticles, frames[(int)frameNum].metabolicParticles.Length, frames[(int)frameNum]);
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
        frames.ForEach(f => {
            f.particles.Dispose();
            f.metabolicParticles.Dispose();
        });
    }
}
