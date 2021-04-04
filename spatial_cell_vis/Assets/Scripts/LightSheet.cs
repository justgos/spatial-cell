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


public class LightSheet : MonoBehaviour
{
    public Slider frameSlider;
    public Text frameNumberText;

    public ComputeShader frameConverter;

    private float lastFrameTime = 0;
    private bool playing = false;
    private float playbackFps = 15;

    private ushort[] dataShape;
    private LRUDictionary<int, RenderTexture> frames;
    private long frameSize;
    private List<long> frameOffsets = new List<long>();
    private Texture currentFrame;

    void Start()
    {
        IndexFrames();
    }

    void IndexFrames()
    {
        System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
        sw.Start();

        using (var fs = File.OpenRead(@"../../data/light sheet/converted.dat"))
        {
            var br = new BinaryReader(fs);
            dataShape = new ushort[5];
            for (var i = 0; i < 5; i++)
                dataShape[i] = br.ReadUInt16();

            var maxDim = (float)Mathf.Max(dataShape[2], dataShape[3], dataShape[4]);
            transform.localScale = new Vector3(
                dataShape[4] / maxDim,
                dataShape[3] / maxDim,
                1.0f // dataShape[2] / maxDim
            );

            frameSize = (long)dataShape[2] * dataShape[3] * dataShape[4] * sizeof(short);
            Debug.Log(string.Format("Frame size: {0}", frameSize));

            long memoryLimit = 8L * 1024 * 1024 * 1024; // 8GB
            var numCachedFrames = (int)Math.Max(memoryLimit / frameSize, 1);
            frames = new LRUDictionary<int, RenderTexture>(
                numCachedFrames,
                (f) =>
                {
                    //f.Dispose();
                }
            );

            for(var i = 0L; i < dataShape[1]; i++)
            {
                frameOffsets.Add(br.BaseStream.Position + i * (frameSize));
            }

            frameSlider.maxValue = dataShape[1] - 1;
            ChangeFrame(0);
        }

        sw.Stop();
        Debug.Log("Frames indexed in " + ((double)sw.ElapsedTicks / System.Diagnostics.Stopwatch.Frequency) + "s");
    }

    async Task<RenderTexture> LoadFrame(int idx)
    {
        System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
        sw.Start();

        var frame = new RenderTexture(dataShape[4], dataShape[3], 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
        frame.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
        frame.volumeDepth = dataShape[2];
        frame.enableRandomWrite = true;
        frame.Create();

        using (var fs = File.OpenRead(@"../../data/light sheet/converted.dat"))
        {
            var br = new BinaryReader(fs);

            br.BaseStream.Position = frameOffsets[idx];
            var bytes = new byte[frameSize];
            await br.BaseStream.ReadAsync(bytes, 0, (int)frameSize);

            var rawFrame = new NativeArray<int>((int)(frameSize / sizeof(int)), Allocator.Persistent);

            unsafe
            {
                fixed (void* bytesPointer = bytes)
                {
                    UnsafeUtility.MemCpy(rawFrame.GetUnsafePtr(), bytesPointer, frameSize);
                }
            }

            //unsafe
            //{
            //    for (var i = 0; i < frameSize / sizeof(int); i++) {
            //        var p = UnsafeUtility.ReadArrayElement<int>(uintFrame.GetUnsafeReadOnlyPtr(), i);
            //        if (p > 1000)
            //        {
            //            Debug.Log(string.Format("Non-zero pixel at {0}: {1}", i, p));
            //            break;
            //        }
            //    }
            //}

            var arrayBuffer = new ComputeBuffer((int)(frameSize / sizeof(int)), sizeof(int));
            arrayBuffer.SetData(rawFrame);
            frameConverter.SetBuffer(0, "array", arrayBuffer);
            frameConverter.SetTexture(0, "tex", frame);
            frameConverter.SetInt("w", dataShape[4]);
            frameConverter.SetInt("h", dataShape[3]);
            frameConverter.SetInt("d", dataShape[2]);
            frameConverter.Dispatch(0, dataShape[4] / 8, dataShape[3] / 8, dataShape[2] / 8);

            arrayBuffer.Dispose();
            rawFrame.Dispose();

            
        }

        sw.Stop();
        Debug.Log(string.Format("Frame {0} loaded in {1}s", idx, ((double)sw.ElapsedTicks / System.Diagnostics.Stopwatch.Frequency)));

        return frame;
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
        if (!gameObject.activeInHierarchy)
            return;
        var frameNum = (int)frameNumF;
        var frame = frames.get(frameNum);
        if (frame == null)
        {
            frame = await LoadFrame(frameNum);
            frames.add(frameNum, frame);
        }

        currentFrame = frame;

        frameNumberText.text = frameSlider.value.ToString();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
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
            if (this.lastFrameTime + 1.0f / this.playbackFps <= Time.time)
            {
                this.lastFrameTime = Time.time;
                SetFrame(frameSlider.value + 1);
            }
        }


        Material material = GetComponent<Renderer>().material;
        material.SetTexture("_Data", currentFrame);
        material.SetVector("_DataChannel", new Vector4(0, 0, 0, 1));
    }
    void OnDestroy()
    {
        if (frames != null)
        {
            foreach (var f in frames)
            {
                //f.value.Dispose();
            }
        }
    }
}
