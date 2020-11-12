using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class SimData : MonoBehaviour
{
    public GameObject spherePrefab;

    public class SimFrame
    {
        public Vector3[] positions;
    }

    private List<SimFrame> frames = new List<SimFrame>();
    public ComputeBuffer frameBuffer;

    void Start()
    {
        var simFrames = File.ReadAllBytes(@"../spatial_cell_sim/results/frames.dat");
        var br = new BinaryReader(new MemoryStream(simFrames));
        var nParticles = 1 * 1024 * 1024;
        var nFrames = 3;
        frameBuffer = new ComputeBuffer(nParticles, sizeof(float) * 3);
        for (var j = 0; j < nFrames; j++)
        {
            var frame = new SimFrame();
            frame.positions = new Vector3[nParticles];
            for (var i = 0; i < nParticles; i++)
            {
                var pos = new Vector3(
                    br.ReadSingle(),
                    br.ReadSingle(),
                    br.ReadSingle()
                );
                frame.positions[i] = pos;
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
        frameBuffer.SetData(frames[0].positions);
    }

    void Update()
    {
        
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
