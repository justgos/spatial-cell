using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class MathDebug : MonoBehaviour
{
    public GameObject debugObject;
    private List<GameObject> debugObjects = new List<GameObject>();
    private float lastGenerationTime = 0;

    public GameObject refObjectA;
    public GameObject refObjectB;
    public GameObject refObjectC;
    private GameObject reverseObjectA;
    private GameObject reverseObjectB;

    void Start()
    {
        if (refObjectA != null && refObjectA.activeSelf)
        {
            reverseObjectA = Instantiate(refObjectB, refObjectB.transform.position, refObjectB.transform.rotation);
            reverseObjectA.name = "reverseObjectA";
            reverseObjectA.GetComponent<Renderer>().material.color = Color.blue;
            reverseObjectB = Instantiate(refObjectA, refObjectA.transform.position, refObjectA.transform.rotation);
            reverseObjectB.name = "reverseObjectB";
            reverseObjectB.GetComponent<Renderer>().material.color = Color.green;
        }
    }

    void Update()
    {
        if(debugObject != null && debugObject.activeSelf && lastGenerationTime + 1.0f < Time.time)
        {
            debugObjects.ForEach(o => GameObject.Destroy(o));

            var targetRelativeOrientation = Quaternion.AngleAxis(30, Vector3.forward);
            var targetRelativePositionRotation = Quaternion.AngleAxis(-30, Vector3.right);
            var prevObject = debugObject;
            var interactionDistance = 1.5f;
            for (var i = 0; i < 20; i++)
            {
                var debugObjectA = Instantiate(prevObject);
                debugObjectA.name = "debugObjectA " + i;
                debugObjectA.transform.rotation = prevObject.transform.rotation * Quaternion.Inverse(targetRelativeOrientation);
                debugObjectA.transform.position = prevObject.transform.position + -((prevObject.transform.rotation * Quaternion.Inverse(targetRelativeOrientation) * targetRelativePositionRotation) * (interactionDistance * Vector3.up));
                debugObjectA.GetComponent<Renderer>().material.color = Color.green;
                debugObjects.Add(debugObjectA);

                var debugObjectB = Instantiate(prevObject);
                debugObjectB.name = "debugObjectB " + i;
                debugObjectB.transform.rotation = debugObjectA.transform.rotation * (targetRelativeOrientation);
                debugObjectB.transform.position = debugObjectA.transform.position + ((debugObjectA.transform.rotation * targetRelativePositionRotation) * (interactionDistance * Vector3.up));
                debugObjectB.transform.localScale *= 1.1f;
                debugObjectB.GetComponent<Renderer>().material.color = Color.blue;
                debugObjects.Add(debugObjectB);

                prevObject = debugObjectA;
            }

            lastGenerationTime = Time.time;
        }

        if(refObjectA != null && refObjectA.activeSelf)
        {
            var up = refObjectC.transform.rotation * Vector3.up;
            float targetRelativePositionAngle = Mathf.PI / 2;
            int n = 0;
            new List<GameObject> { refObjectA, refObjectB }.ForEach(go =>
            {
                var tup = go.transform.rotation * Vector3.up;
                var delta = go.transform.position - refObjectC.transform.position;
                var normalizedDelta = delta.normalized;
                var upDeltaQ = Quaternion.FromToRotation(up, normalizedDelta);
                float upDeltaAngle;
                Vector3 upDeltaAxis;
                upDeltaQ.ToAngleAxis(out upDeltaAngle, out upDeltaAxis);
                //if (n == 0)
                //{
                //    Debug.Log("upDeltaAngle " + upDeltaAngle + ", upDeltaAxis " + upDeltaAxis);
                //    reverseObjectA.transform.position = refObjectC.transform.position + (Quaternion.AngleAxis(targetRelativePositionAngle - upDeltaAngle, Vector3.Cross(up, normalizedDelta)) * Vector3.up);
                //}
                //else
                //    reverseObjectB.transform.position = refObjectC.transform.position + (Quaternion.AngleAxis(targetRelativePositionAngle - 2.0f * Mathf.Acos(Quaternion.FromToRotation(up, normalizedDelta).w), Vector3.Cross(up, normalizedDelta)) * Vector3.up);
                refObjectC.transform.rotation = Quaternion.Slerp(
                    refObjectC.transform.rotation,
                    Quaternion.FromToRotation(
                        up,
                        Quaternion.AngleAxis(90, Vector3.Cross(normalizedDelta, up)) * normalizedDelta
                    ) * refObjectC.transform.rotation,
                    //Quaternion.AngleAxis(targetRelativePositionAngle - 2.0f * Mathf.Acos(Quaternion.FromToRotation(up, normalizedDelta).w), Vector3.Cross(up, normalizedDelta)) * refObjectC.transform.rotation,
                    0.05f
                );
                refObjectC.transform.rotation = Quaternion.Slerp(
                    refObjectC.transform.rotation,
                    Quaternion.FromToRotation(up, tup) * refObjectC.transform.rotation,
                    0.05f
                );
                n++;
            });
            
            //reverseObjectA.transform.rotation = refObjectA.transform.rotation;
            //reverseObjectA.transform.position = Quaternion.AngleAxis(
            //    targetRelativePositionAngle,
            //    Vector3.Cross(refObjectA.transform.rotation * Vector3.up, -(refObjectA.transform.position - reverseObjectA.transform.position)).normalized
            //) * (reverseObjectA.transform.rotation * Vector3.up) + refObjectA.transform.position;
            ////reverseObjectA.transform.position = refObjectA.transform.position + Vector3.forward;

            //reverseObjectB.transform.rotation = refObjectB.transform.rotation;
            //reverseObjectB.transform.position = Quaternion.AngleAxis(
            //    targetRelativePositionAngle,
            //    Vector3.Cross(refObjectB.transform.rotation * Vector3.up, -(refObjectB.transform.position - reverseObjectB.transform.position).normalized)
            //) * (reverseObjectB.transform.rotation * Vector3.up) + refObjectB.transform.position;
            ////reverseObjectB.transform.position = Vector3.Cross(refObjectA.transform.rotation * Vector3.up, -(refObjectA.transform.position - reverseObjectA.transform.position).normalized) + refObjectA.transform.position;
            ////Debug.Log("Cross " + Vector3.Cross(refObjectB.transform.rotation * Vector3.up, -(refObjectB.transform.position - reverseObjectA.transform.position).normalized).ToString("F4"));
        }
    }
}
