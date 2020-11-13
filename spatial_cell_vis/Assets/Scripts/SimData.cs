using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;

public class SimData : MonoBehaviour
{
    public GameObject spherePrefab;
    public struct Particle
    {
        public Vector3 pos;
        public Vector3 velocity;
        public int type;
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
    public ComputeBuffer frameBuffer;

    public Slider frameSlider;

    void Start()
    {
        var simFrames = File.ReadAllBytes(@"../spatial_cell_sim/results/frames.dat");
        var br = new BinaryReader(new MemoryStream(simFrames));
        simSize = br.ReadSingle();
        particleBufferSize = br.ReadInt32();
        frameBuffer = new ComputeBuffer(particleBufferSize, Marshal.SizeOf(new Particle()));
        while(br.BaseStream.Position != br.BaseStream.Length)
        //for (var j = 0; j < nFrames; j++)
        {
            var frame = new SimFrame();
            frame.numParticles = br.ReadUInt32();
            frame.particles = new Particle[frame.numParticles];
            for (var i = 0; i < frame.numParticles; i++)
            {
                var p = new Particle();
                p.pos = new Vector3(
                    br.ReadSingle(),
                    br.ReadSingle(),
                    br.ReadSingle()
                );
                p.velocity = new Vector3(
                    br.ReadSingle(),
                    br.ReadSingle(),
                    br.ReadSingle()
                );
                p.type = br.ReadInt32();
                frame.particles[i] = p;
                //Debug.Log("pos " + pos.ToString("F4"));
                //var sphere = Instantiate(spherePrefab, pos, Quaternion.identity);
                //if(j == 0)
                //    sphere.GetComponent<Renderer>().material.color = Color.red;
                //else if (j == 1)
                //    sphere.GetComponent<Renderer>().material.color = Color.green;
                //else if (j == 2)
                //    sphere.GetComponent<Renderer>().material.color = Color.blue;
            }
            frames.Add(frame);
        }
        frameSlider.maxValue = frames.Count - 1;
        frameBuffer.SetData(frames[0].particles);
    }

    void Update()
    {
        //
    }

    public void ChangeFrame(float frameNum)
    {
        frameBuffer.SetData(frames[(int)frameNum].particles);
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
