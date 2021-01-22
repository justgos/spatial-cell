using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.UI.Extensions;
using Unity.Collections.LowLevel.Unsafe;

public class ParticlePicker : MonoBehaviour
{
    public Camera camera;

    public RangeSlider particleVisibleRangeXSlider;
    public RangeSlider particleVisibleRangeYSlider;
    public RangeSlider particleVisibleRangeZSlider;

    private Particles targetParticleRenderer = null;
    private Nullable<SimData.Particle> targetParticle = null;
    private Nullable<SimData.Particle> foundTargetParticle = null;

    public ComputeShader findParticles;
    private ComputeBuffer foundParticles;
    private ComputeBuffer foundParticlesArgs;
    public Text targetParticleInfo;

    void Start()
    {
        foundParticles = new ComputeBuffer(1, Marshal.SizeOf(typeof(SimData.Particle)), ComputeBufferType.Append);
        foundParticlesArgs = new ComputeBuffer(5, sizeof(int), ComputeBufferType.IndirectArguments);
    }

    // Ref: http://paulbourke.net/geometry/circlesphere/raysphere.c
    bool IntersectRaySphere(Ray r, Vector3 sphereCenter, float sphereRadius)
    {
        float a, b, c;
        float bb4ac;
        Vector3 dp = r.direction;

        a = dp.x * dp.x + dp.y * dp.y + dp.z * dp.z;
        b = 2 * (dp.x * (r.origin.x - sphereCenter.x) + dp.y * (r.origin.y - sphereCenter.y) + dp.z * (r.origin.z - sphereCenter.z));
        c = sphereCenter.x * sphereCenter.x + sphereCenter.y * sphereCenter.y + sphereCenter.z * sphereCenter.z;
        c += r.origin.x * r.origin.x + r.origin.y * r.origin.y + r.origin.z * r.origin.z;
        c -= 2 * (sphereCenter.x * r.origin.x + sphereCenter.y * r.origin.y + sphereCenter.z * r.origin.z);
        c -= sphereRadius * sphereRadius;
        bb4ac = b * b - 4 * a * c;
        if (Mathf.Abs(a) < 1e-6 || bb4ac < 0)
        {
            return false;
        }

        return true;
    }

    void Update()
    {
        if (Input.GetButtonDown("Fire1") && !EventSystem.current.IsPointerOverGameObject())
        { 
            Ray ray = camera.ScreenPointToRay(Input.mousePosition);
            targetParticleRenderer = null;
            targetParticle = null;
            int targetParticleIdx = -1;
            float targetParticleDist = float.MaxValue;
            var particleRenderers = GameObject.FindObjectsOfType<Particles>();
            foreach (var particleRenderer in particleRenderers)
            {
                if (!particleRenderer.gameObject.activeSelf || particleRenderer.frameData.Frame == null)
                    continue;
                SimData.SimFrame frame = (SimData.SimFrame)particleRenderer.frameData.Frame;
                unsafe
                {
                    for (var i = 0; i < frame.particles.Length; i++)
                    {
                        var p = UnsafeUtility.ReadArrayElement<SimData.Particle>(frame.particles.GetUnsafeReadOnlyPtr(), i);

                        if (
                            (p.flags & SimData.PARTICLE_FLAG_ACTIVE) < 1
                            || p.pos.x < particleVisibleRangeXSlider.LowValue
                            || p.pos.x > particleVisibleRangeXSlider.HighValue
                            || p.pos.y < particleVisibleRangeYSlider.LowValue
                            || p.pos.y > particleVisibleRangeYSlider.HighValue
                            || p.pos.z < particleVisibleRangeZSlider.LowValue
                            || p.pos.z > particleVisibleRangeZSlider.HighValue
                        ) {
                            continue;
                        }
                        var pPos = p.pos.UnityVector();
                        var particlePos = particleRenderer.transform.position + pPos * particleRenderer.DrawScale;
                        //Debug.Log("particlePos " + p.id + ", " + particlePos.ToString("F4") + ", r " + p.radius * 10);
                        if (IntersectRaySphere(ray, particlePos, p.radius * particleRenderer.DrawScale))
                        {
                            var particleDist = (ray.origin - particlePos).magnitude;
                            if (particleDist < targetParticleDist)
                            {
                                targetParticleRenderer = particleRenderer;
                                targetParticleIdx = i;
                                targetParticle = p;
                                targetParticleDist = particleDist;
                            }
                        }
                    }
                    
                    //fixed (SimData.Particle* ps = particleRenderer.frameData.Frame.particles.GetUnsafeReadOnlyPtr())
                    //{
                    //    Debug.Log("ps " + m[0] + ", " + m[1] + ", " + m[2] + ", " + m[3]);
                    //}
                }
            }

            if(targetParticle != null)
            {
                SimData.Particle tp = (SimData.Particle)targetParticle;
                //LogParticleState(tp.id);

                foreach (var particleRenderer in particleRenderers)
                {
                    particleRenderer.TargetParticleId = tp.id;
                }
            } else {
                foreach (var particleRenderer in particleRenderers)
                {
                    particleRenderer.TargetParticleId = -1;
                }
            }
        }

        ResolveTargetParticle();

        //if (Input.GetKeyDown(KeyCode.X))
        //{
        //    if (targetParticle != null)
        //    {
        //        SimData.Particle tp = (SimData.Particle)targetParticle;

            //        LogParticleState(tp.id);
            //    }
            //}
    }

    Nullable<SimData.Particle> FindParticle(ComputeBuffer particleBuffer, int numParticles, int id)
    {
        Nullable<SimData.Particle> target = null;
        foundParticles.SetCounterValue(0);
        findParticles.SetBuffer(0, "particles", particleBuffer);
        findParticles.SetBuffer(0, "foundParticles", foundParticles);
        findParticles.SetInt("particleId", id);
        findParticles.SetInt("count", numParticles);
        findParticles.Dispatch(0, particleBuffer.count / 8, 1, 1);
        int[] args = new int[] { 0, 1, 0, 0, 0 };
        foundParticlesArgs.SetData(args);
        ComputeBuffer.CopyCount(foundParticles, foundParticlesArgs, 0);
        foundParticlesArgs.GetData(args);
        //Debug.Log(string.Format("Found {0} particles with id {1}", args[0], id));
        if(args[0] > 0)
        {
            var found = new SimData.Particle[1];
            foundParticles.GetData(found);
            target = found[0];
        }

        return target;
    }

    void ResolveTargetParticle()
    {
        if(!targetParticle.HasValue)
        {
            foundTargetParticle = null;
            targetParticleInfo.text = "";
            return;
        }

        var particleRenderers = GameObject.FindObjectsOfType<Particles>();
        foreach (var particleRenderer in particleRenderers)
        {
            if (!particleRenderer.gameObject.activeSelf || particleRenderer.frameData.Frame == null)
                continue;
            var found = FindParticle(particleRenderer.frameData.ParticleBuffer, particleRenderer.frameData.NumParticles, targetParticle.Value.id);
            foundTargetParticle = found;
            if (foundTargetParticle.HasValue)
            {
                var p = foundTargetParticle.Value;
                targetParticleInfo.text = string.Format("Target particle: \nid {0}, \ntype {1}, \nflags {2:X}, \nstate {3}, \nr {4}, \npos {5}, \nrot {6}, \ndebugVector {7}",
                    p.id,
                    p.type,
                    p.flags & 0xFFFF,
                    (p.flags >> 16) & 0xFFFF,
                    p.radius,
                    p.pos.UnityVector().ToString("F4"),
                    p.rot.UnityQuaternion().ToString("F4"),
                    p.debugVector.UnityQuaternion().ToString("F6")
                );
                return;
            }
            else
            {
                targetParticleInfo.text = "Target particle: \nNot found";
            }
        }
    }

    void LogParticleState(int id)
    {
        var particleRenderers = GameObject.FindObjectsOfType<Particles>();
        foreach (var particleRenderer in particleRenderers)
        {
            if (!particleRenderer.gameObject.activeSelf || particleRenderer.frameData.Frame == null)
                continue;
            SimData.SimFrame frame = (SimData.SimFrame)particleRenderer.frameData.Frame;
            unsafe
            {
                var found = FindParticle(particleRenderer.frameData.ParticleBuffer, particleRenderer.frameData.NumParticles, id);
                if(found.HasValue)
                {
                    var p = found.Value;
                    Debug.Log(string.Format("Target particle: id {0}, type {1}, flags {2:X}, pos {3}, rot {4}, r {5}, debugVector {6}",
                        p.id,
                        p.type,
                        p.flags,
                        p.pos.UnityVector().ToString("F4"),
                        p.rot.UnityQuaternion().ToString("F4"),
                        p.radius,
                        p.debugVector.UnityQuaternion().ToString("F6")
                    ));
                }

                //for (var i = 0; i < frame.particles.Length; i++)
                //{
                //    var p = UnsafeUtility.ReadArrayElement<SimData.Particle>(frame.particles.GetUnsafeReadOnlyPtr(), i);

                //    if (p.id == id)
                //    {
                //        Debug.Log(string.Format("Target particle: id {0}, type {1}, flags {2:X}, pos {3}, rot {4}, r {5}, debugVector {6}",
                //            p.id,
                //            p.type,
                //            p.flags,
                //            p.pos.UnityVector().ToString("F4"),
                //            p.rot.UnityQuaternion().ToString("F4"),
                //            p.radius,
                //            p.debugVector.UnityQuaternion().ToString("F6")
                //        ));

                //        return;
                //    }
                //}
            }
        }
    }

    void OnDestroy()
    {
        foundParticles.Dispose();
        foundParticlesArgs.Dispose();
    }
}
