using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
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

    void Start()
    {
        
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
                        var particlePos = particleRenderer.transform.position + pPos * 10;
                        //Debug.Log("particlePos " + p.id + ", " + particlePos.ToString("F4"));
                        if (IntersectRaySphere(ray, particlePos, p.radius * 10))
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
                Debug.Log("Target particle " + tp.id);

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

        if (Input.GetKeyDown(KeyCode.X))
        {
            if (targetParticle != null)
            {
                SimData.Particle tp = (SimData.Particle)targetParticle;

                LogParticleState(tp.id);
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
                for (var i = 0; i < frame.particles.Length; i++)
                {
                    var p = UnsafeUtility.ReadArrayElement<SimData.Particle>(frame.particles.GetUnsafeReadOnlyPtr(), i);

                    if (p.id == id)
                    {
                        Debug.Log(string.Format("Target particle: id {0}, type {1}, pos {2}, rot {3}, debugVector {4}",
                            p.id,
                            p.type,
                            p.pos.UnityVector().ToString("F4"),
                            p.rot.UnityQuaternion().ToString("F4"),
                            p.debugVector.UnityQuaternion().ToString("F6")
                        ));

                        return;
                    }
                }
            }
        }
    }
}
