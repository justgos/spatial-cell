using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
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

    [StructLayout(LayoutKind.Explicit, Size = 48)]
    public struct Particle
    {
        //[MarshalAs(UnmanagedType.LPStruct)]
        [FieldOffset(0)]  public Vector3_ pos;
        [FieldOffset(16)] public Vector4_ rot;
        [FieldOffset(32)] public Vector3_ velocity;
        [FieldOffset(44)] public int type;
    };

    public class SimFrame
    {
        public uint numParticles;
        public Particle[] particles;
    }

    private float simSize;
    public float SimSize { get { return simSize; } }
    private int particleBufferSize;
    private List<SimFrame> frames = new List<SimFrame>();
    private int numParticles;
    public int NumParticles { get { return numParticles; } }
    public ComputeBuffer frameBuffer;

    public Slider frameSlider;

    private float lastFrameTime = 0;
    private bool playing = false;
    private float playbackFps = 30;

    void Start()
    {
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
                frame.particles = new Particle[frame.numParticles];
                var bytes = br.ReadBytes((int)(particleStructSize * frame.numParticles));
                //Marshal.Copy(, 0, (IntPtr)frame.particles, 0, (int)(particleStructSize * frame.numParticles));
                for (var i = 0; i < frame.numParticles; i++)
                    frame.particles[i] = Marshal.PtrToStructure<Particle>(Marshal.UnsafeAddrOfPinnedArrayElement(bytes, particleStructSize * i));
                //for (var i = 0; i < frame.numParticles; i++)
                //{
                //    var p = new Particle();
                //    p.pos = new Vector3(
                //        br.ReadSingle(),
                //        br.ReadSingle(),
                //        br.ReadSingle()
                //    );
                //    p.velocity = new Vector3(
                //        br.ReadSingle(),
                //        br.ReadSingle(),
                //        br.ReadSingle()
                //    );
                //    p.type = br.ReadInt32();
                //    frame.particles[i] = p;
                //    //Debug.Log("pos " + pos.ToString("F4"));
                //    //var sphere = Instantiate(spherePrefab, pos, Quaternion.identity);
                //    //if(j == 0)
                //    //    sphere.GetComponent<Renderer>().material.color = Color.red;
                //    //else if (j == 1)
                //    //    sphere.GetComponent<Renderer>().material.color = Color.green;
                //    //else if (j == 2)
                //    //    sphere.GetComponent<Renderer>().material.color = Color.blue;
                //}
                frames.Add(frame);
            }
            frameSlider.maxValue = frames.Count - 1;
            frameBuffer.SetData(frames[0].particles);
            numParticles = frames[0].particles.Length;
        }
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
            SetFrame(frameSlider.value - 1);
        }
        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            SetFrame(frameSlider.value + 1);
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
    }
}
