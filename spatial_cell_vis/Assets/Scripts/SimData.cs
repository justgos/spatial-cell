using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.UI;
using UnityEngine.UI.Extensions;
using LRUDictionary;


public class SimData : MonoBehaviour
{
    public static readonly int PARTICLE_FLAG_ACTIVE = 0x0001;

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
    public struct VectorHalf3_
    {
        //[MarshalAs(UnmanagedType.fl)]
        public Half x;
        public Half y;
        public Half z;

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
    public struct VectorHalf4_
    {
        public Half x;
        public Half y;
        public Half z;
        public Half w;

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

    //[StructLayout(LayoutKind.Explicit, Size = 144)]
    //unsafe public struct Particle
    //{
    //    [FieldOffset(0)]  public int id;
    //    [FieldOffset(4)]  public int type;
    //    [FieldOffset(8)]  public int flags;
    //    [FieldOffset(12)] public Vector3_ pos;
    //    [FieldOffset(24)] public fixed float __padding1[2];
    //    [FieldOffset(32)] public Vector4_ rot;
    //    [FieldOffset(48)] public Vector3_ velocity;
    //    [FieldOffset(64)] public Vector4_ angularVelocity;
    //    [FieldOffset(80)] public int nActiveInteractions;
    //    [FieldOffset(84)] public ParticleInteraction interaction1;
    //    [FieldOffset(92)] public ParticleInteraction interaction2;
    //    [FieldOffset(100)] public ParticleInteraction interaction3;
    //    [FieldOffset(108)] public ParticleInteraction interaction4;
    //    [FieldOffset(128)] public Vector4_ debugVector;
    //};

    // ReducedParticle
    [StructLayout(LayoutKind.Explicit, Size = 28)]  // 48
    unsafe public struct Particle
    {
        [FieldOffset(0)] public int id;
        [FieldOffset(4)] public int type;
        [FieldOffset(8)] public int flags;
        [FieldOffset(12)] public Half radius;
        [FieldOffset(14)] public VectorHalf3_ pos;
        [FieldOffset(20)] public VectorHalf4_ rot;
        //[FieldOffset(28)] public VectorHalf4_ debugVector;
        //[FieldOffset(128)] public Vector4_ debugVector;
    };

    //[StructLayout(LayoutKind.Explicit, Size = 352)]
    //unsafe public struct MetabolicParticle
    //{
    //    [FieldOffset(0)] public int id;
    //    [FieldOffset(4)] public int type;
    //    [FieldOffset(8)] public int flags;
    //    [FieldOffset(12)] public Vector3_ pos;
    //    [FieldOffset(24)] public fixed float __padding1[2];
    //    [FieldOffset(32)] public Vector4_ rot;
    //    [FieldOffset(48)] public Vector3_ velocity;
    //    [FieldOffset(64)] public Vector4_ angularVelocity;
    //    [FieldOffset(80)] public int nActiveInteractions;
    //    [FieldOffset(84)] public ParticleInteraction interaction1;
    //    [FieldOffset(92)] public ParticleInteraction interaction2;
    //    [FieldOffset(100)] public ParticleInteraction interaction3;
    //    [FieldOffset(108)] public ParticleInteraction interaction4;
    //    [FieldOffset(128)] public Vector4_ debugVector;
    //    [FieldOffset(144)] public fixed float metabolites[50];
    //};

    // ReducedMetabolicParticle
    [StructLayout(LayoutKind.Explicit, Size = 36)]  // 64
    unsafe public struct MetabolicParticle
    {
        [FieldOffset(0)] public int id;
        [FieldOffset(4)] public int type;
        [FieldOffset(8)] public int flags;
        [FieldOffset(12)] public Half radius;
        [FieldOffset(14)] public VectorHalf3_ pos;
        [FieldOffset(20)] public VectorHalf4_ rot;
        //[FieldOffset(28)] public VectorHalf4_ debugVector;
        //[FieldOffset(128)] public Vector4_ debugVector;
        //[FieldOffset(48)] public fixed float metabolites[4];
    };

    public struct SimFrame
    {
        public uint numParticles;
        //public Particle[] particles;
        public NativeArray<Particle> particles;
        public Dictionary<int, NativeList<Particle>> particleMap;
        public uint numMetabolicParticles;
        public NativeArray<MetabolicParticle> metabolicParticles;
    }

    private float simSize;
    public float SimSize { get { return simSize; } }
    private LRUDictionary<int, Nullable<SimFrame>> frames;
    private List<long> frameOffsets = new List<long>();

    int particleBufferSize;
    int particleStructSize;
    int metabolicParticleBufferSize;
    int metabolicParticleStructSize;

    public FrameData baseParticleFrameData;
    private SortedDictionary<int, FrameData> particleFrameData = new SortedDictionary<int, FrameData>();
    public FrameData metabolicParticleFrameData;

    public Particles baseParticleRenderer;
    private SortedDictionary<int, Particles> particleRenderers = new SortedDictionary<int, Particles>();
    public Particles metabolicParticleRenderer;

    // Lipid bilayer model
    // Ref: https://www.umass.edu/microbio/rasmol/bilayers.htm
    public GameObject particleType0Model;

    public Slider frameSlider;
    public Text frameNumberText;

    private float lastFrameTime = 0;
    private bool playing = false;
    private float playbackFps = 15;

    public RangeSlider particleVisibleRangeXSlider;
    public RangeSlider particleVisibleRangeYSlider;
    public RangeSlider particleVisibleRangeZSlider;

    void Start()
    {
        baseParticleFrameData.gameObject.SetActive(false);
        baseParticleRenderer.gameObject.SetActive(false);
        metabolicParticleRenderer.gameObject.SetActive(false);

        IndexFrames();

        // Enable the metabolic renderer after the plain ones,
        // so that it's drawn on top
        metabolicParticleRenderer.gameObject.SetActive(true);
    }

    void IndexFrames()
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

            particleBufferSize = br.ReadInt32();
            particleStructSize = Marshal.SizeOf(typeof(Particle));
            Debug.Log("numParticles " + particleBufferSize);
            Debug.Log("particleStructSize " + particleStructSize);
            //particleFrameData.Init(particleBufferSize, particleStructSize);
            metabolicParticleBufferSize = br.ReadInt32();
            metabolicParticleStructSize = Marshal.SizeOf(typeof(MetabolicParticle));
            Debug.Log("numMetabolicParticles " + metabolicParticleBufferSize);
            Debug.Log("metabolicParticleStructSize " + metabolicParticleStructSize);
            //metabolicParticleFrameData.Init(metabolicParticleBufferSize, metabolicParticleStructSize);

            long memoryLimit = 8L * 1024 * 1024 * 1024; // 8GB
            var numCachedFrames = (int)Math.Max(memoryLimit / ((long)particleBufferSize * particleStructSize + (long)metabolicParticleBufferSize * metabolicParticleStructSize), 1);
            Debug.Log("numCachedFrames " + numCachedFrames);

            frames = new LRUDictionary<int, Nullable<SimFrame>>(
                numCachedFrames,
                (f) => {
                    f.Value.particles.Dispose();
                    foreach (var entry in f.Value.particleMap)
                        entry.Value.Dispose();
                    f.Value.metabolicParticles.Dispose();
                }
            );

            var particleMaxCountByType = new SortedDictionary<int, int>();
            particleMaxCountByType.Add(0, particleBufferSize);

            while (br.BaseStream.Position != br.BaseStream.Length)
            {
                //var frame = new SimFrame();
                var numParticles = br.ReadUInt32();
                //frame.numParticles = br.ReadUInt32();
                if (numParticles < 1)
                    break;
                //frame.particles = new NativeArray<Particle>((int)frame.numParticles, Allocator.Persistent);
                int frameSize = (int)(particleStructSize * numParticles);
                frameOffsets.Add(br.BaseStream.Position - sizeof(uint));
                //var bytes = br.ReadBytes(frameSize);

                br.BaseStream.Position += frameSize;

                var numMetabolicParticles = br.ReadUInt32();
                frameSize = (int)(metabolicParticleStructSize * numMetabolicParticles);

                br.BaseStream.Position += frameSize;

                //unsafe
                //{
                //    //byte* ptr = (byte*)0;
                //    //accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
                //    fixed (void* bytesPointer = bytes)
                //    {
                //        //UnsafeUtility.CopyStructureToPtr((byte*)bytes[0], frame.particles.GetUnsafePtr());
                //        UnsafeUtility.MemCpy(frame.particles.GetUnsafePtr(), bytesPointer, UnsafeUtility.SizeOf<Particle>() * frame.numParticles);
                //        //UnsafeUtility.MemCpy(frame.particles.GetUnsafePtr(), ptr + br.BaseStream.Position, UnsafeUtility.SizeOf<Particle>() * frame.numParticles);
                //    }
                //    //accessor.SafeMemoryMappedViewHandle.ReleasePointer();


                //    //fixed (Particle* bytesPointer = frame.particles.ToArray())
                //    //{
                //    //    NativeList<Particle> typeList;
                //    //    frame.particleMap.TryGetValue(p.type, out typeList);
                //    //}
                //}
                ////br.BaseStream.Position += frameSize;

                //frame.particleMap = new Dictionary<int, NativeList<Particle>>();
                //foreach (var p in frame.particles.ToArray())
                //{
                //    //Debug.Log("p " + p.id + ", " + p.pos.UnityVector().ToString("F4"));
                //    NativeList<Particle> typeList;
                //    if (!frame.particleMap.ContainsKey(p.type))
                //    {
                //        typeList = new NativeList<Particle>(0, Allocator.Persistent);
                //        frame.particleMap.Add(p.type, typeList);
                //    }
                //    else
                //    {
                //        typeList = frame.particleMap[p.type];
                //    }
                //    typeList.Add(p);
                //}

                //foreach(var entry in frame.particleMap)
                //{
                //    int particleCount = 0;
                //    particleCount = particleMaxCountByType.TryGetValue(entry.Key, out particleCount) ? particleCount : 0;
                //    particleCount = Math.Max(particleCount, entry.Value.Length);
                //    particleMaxCountByType[entry.Key] = particleCount;
                //}

                //frame.numMetabolicParticles = br.ReadUInt32();
                //frame.metabolicParticles = new NativeArray<MetabolicParticle>((int)frame.numMetabolicParticles, Allocator.Persistent);
                //frameSize = (int)(metabolicParticleStructSize * frame.numMetabolicParticles);
                //bytes = br.ReadBytes(frameSize);

                //unsafe
                //{
                //    fixed (void* bytesPointer = bytes)
                //    {
                //        UnsafeUtility.MemCpy(frame.metabolicParticles.GetUnsafePtr(), bytesPointer, UnsafeUtility.SizeOf<MetabolicParticle>() * frame.numMetabolicParticles);
                //    }
                //}

                ////UnsafeUtility.
                ////NativeArray.
                ////for (var i = 0; i < frame.numParticles; i++)
                ////    frame.particles[i] = Marshal.PtrToStructure<Particle>(Marshal.UnsafeAddrOfPinnedArrayElement(bytes, particleStructSize * i));

                //frames.Add(frame);

                ////unsafe
                ////{
                ////    fixed (float* m = frame.metabolicParticles.ToArray()[0].metabolites)
                ////    {
                ////        Debug.Log("frame.metabolicParticles " + m[0] + ", " + m[1] + ", " + m[2] + ", " + m[3]);
                ////    }
                ////}
                ////return;
            }

            foreach(var entry in particleMaxCountByType)
            {
                Debug.Log(string.Format("Max count of type {0} particles: {1}", entry.Key, entry.Value));
                var frameData = Instantiate(baseParticleFrameData.gameObject, baseParticleFrameData.transform.parent).GetComponent<FrameData>();
                frameData.gameObject.name = string.Format("{0}-{1}", baseParticleFrameData.gameObject.name, entry.Key);
                //frameData.Init(entry.Value, particleStructSize);
                frameData.gameObject.SetActive(true);
                particleFrameData.Add(entry.Key, frameData);

                var particleRenderer = Instantiate(baseParticleRenderer.gameObject, baseParticleRenderer.transform.parent).GetComponent<Particles>();
                particleRenderer.gameObject.name = string.Format("{0}-{1}", baseParticleRenderer.gameObject.name, entry.Key);
                particleRenderer.frameData = particleFrameData[entry.Key];
                if (entry.Key == 0)
                    particleRenderer.SetModel(particleType0Model);
                particleRenderer.gameObject.SetActive(true);
                particleRenderers.Add(entry.Key, particleRenderer);
            }

            frameSlider.maxValue = frameOffsets.Count - 1;
            ChangeFrame(0);
            //}
        }

        sw.Stop();
        Debug.Log("Frames indexed in " + ((double)sw.ElapsedTicks / System.Diagnostics.Stopwatch.Frequency) + "s");
    }

    async Task<SimFrame> LoadFrame(int idx)
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

        var frame = new SimFrame();

        using (var fs = File.OpenRead(@"../spatial_cell_sim/results/frames.dat"))
        //using (var mmf = MemoryMappedFile.OpenExisting("spatial_cell_buf"))
        {
            //using (var accessor = mmf.CreateViewAccessor(0, 2 * 1024 * 1024 * 1023))
            //{
            //var br = new BinaryReader(fs);
            //var memStream = mmf.CreateViewStream();
            var br = new BinaryReader(fs);

            br.BaseStream.Position = frameOffsets[idx];

            //var frame = new SimFrame();
            frame.numParticles = br.ReadUInt32();
            frame.particles = new NativeArray<Particle>((int)frame.numParticles, Allocator.Persistent);
            int frameSize = (int)(particleStructSize * frame.numParticles);

            //var bytes = br.ReadBytes(frameSize);
            var bytes = new byte[frameSize];
            await br.BaseStream.ReadAsync(bytes, 0, frameSize);

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


                //fixed (Particle* bytesPointer = frame.particles.ToArray())
                //{
                //    NativeList<Particle> typeList;
                //    frame.particleMap.TryGetValue(p.type, out typeList);
                //}
            }
            //br.BaseStream.Position += frameSize;

            frame.particleMap = new Dictionary<int, NativeList<Particle>>();
            var particleList = new NativeList<Particle>((int)frame.numParticles, Allocator.Persistent);
            particleList.AddRange(frame.particles);
            frame.particleMap.Add(0, particleList);
            //var p = frame.particles.ToArray()[100];
            //Debug.Log("p " + p.id + ", " + p.pos.UnityVector().ToString("F4"));
            //foreach (var p in frame.particles.ToArray())
            //{
            //    //Debug.Log("p " + p.id + ", " + p.pos.UnityVector().ToString("F4"));
            //    NativeList<Particle> typeList;
            //    if (!frame.particleMap.ContainsKey(p.type))
            //    {
            //        typeList = new NativeList<Particle>(0, Allocator.Persistent);
            //        frame.particleMap.Add(p.type, typeList);
            //    }
            //    else
            //    {
            //        typeList = frame.particleMap[p.type];
            //    }
            //    typeList.Add(p);
            //}

            //foreach (var entry in frame.particleMap)
            //{
            //    int particleCount = 0;
            //    particleCount = particleMaxCountByType.TryGetValue(entry.Key, out particleCount) ? particleCount : 0;
            //    particleCount = Math.Max(particleCount, entry.Value.Length);
            //    particleMaxCountByType[entry.Key] = particleCount;
            //}

            frame.numMetabolicParticles = br.ReadUInt32();
            frame.metabolicParticles = new NativeArray<MetabolicParticle>((int)frame.numMetabolicParticles, Allocator.Persistent);
            frameSize = (int)(metabolicParticleStructSize * frame.numMetabolicParticles);
            //bytes = br.ReadBytes(frameSize);
            bytes = new byte[frameSize];
            await br.BaseStream.ReadAsync(bytes, 0, frameSize);

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

            //frames.Add(frame);

            //unsafe
            //{
            //    fixed (float* m = frame.metabolicParticles.ToArray()[0].metabolites)
            //    {
            //        Debug.Log("frame.metabolicParticles " + m[0] + ", " + m[1] + ", " + m[2] + ", " + m[3]);
            //    }
            //}
            //return;
        }

        sw.Stop();
        Debug.Log(string.Format("Frame {0} loaded in {1}s", idx, ((double)sw.ElapsedTicks / System.Diagnostics.Stopwatch.Frequency)));

        return frame;
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

        //foreach(var entry in particleRenderers)
        //{
        //    if(Input.GetKeyDown(KeyCode.Alpha0 + entry.Key))
        //    {
        //        entry.Value.gameObject.SetActive(!entry.Value.gameObject.activeSelf);
        //    }
        //}

        for(var i = 0; i < 8; i++)
        {
            if (Input.GetKeyDown(KeyCode.Alpha0 + i))
            {
                particleRenderers[0].ToggleParticleType(i);
            }
        }

        // Toggle metabolic particle rendering
        if(Input.GetKeyDown(KeyCode.M))
        {
            metabolicParticleRenderer.gameObject.SetActive(!metabolicParticleRenderer.gameObject.activeSelf);
        }
        // Toggle smooth/descreet metabolic particle rendering
        if (Input.GetKeyDown(KeyCode.N))
        {
            metabolicParticleRenderer.drawInstanced = !metabolicParticleRenderer.drawInstanced;
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
            frameNum = this.frameOffsets.Count + frameNum;
        else if (frameNum >= this.frameOffsets.Count)
            frameNum = frameNum % this.frameOffsets.Count;
        frameSlider.value = frameNum;
    }

    public async void ChangeFrame(float frameNumF)
    {
        var frameNum = (int)frameNumF;
        var frame = frames.get(frameNum);
        if(frame == null)
        {
            frame = await LoadFrame(frameNum);
            frames.add(frameNum, frame);
        }
        foreach(var entry in particleFrameData)
        {
            if (frame.Value.particleMap.ContainsKey(entry.Key))
                entry.Value.SetData<Particle>(frame.Value.particleMap[entry.Key].AsArray(), frame.Value.particleMap[entry.Key].Length, frame);
            else
                entry.Value.SetData<Particle>(null, 0, null);
        }
        //particleFrameData.SetData(frames[(int)frameNum].particles, frames[(int)frameNum].particles.Length, frames[(int)frameNum]);
        metabolicParticleFrameData.SetData<MetabolicParticle>(frame.Value.metabolicParticles, frame.Value.metabolicParticles.Length, frame);
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
        foreach(var f in frames)
        {
            f.value.Value.particles.Dispose();
            foreach (var entry in f.value.Value.particleMap)
                entry.Value.Dispose();
            f.value.Value.metabolicParticles.Dispose();
        }
        //frames.ForEach(f =>
        //{
        //    f.particles.Dispose();
        //    foreach (var entry in f.particleMap)
        //        entry.Value.Dispose();
        //    f.metabolicParticles.Dispose();
        //});
    }
}
