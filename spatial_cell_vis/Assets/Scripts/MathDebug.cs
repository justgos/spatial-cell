using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MathDebug : MonoBehaviour
{
    public GameObject debugObject;
    private List<GameObject> debugObjects = new List<GameObject>();
    private float lastGenerationTime = 0;

    public GameObject refObjectA;
    public GameObject refObjectB;
    private GameObject reverseObjectA;
    private GameObject reverseObjectB;

    void Start()
    {
        reverseObjectA = Instantiate(refObjectB, refObjectB.transform.position, refObjectB.transform.rotation);
        reverseObjectA.name = "reverseObjectA";
        reverseObjectA.GetComponent<Renderer>().material.color = Color.blue;
        reverseObjectB = Instantiate(refObjectA, refObjectA.transform.position, refObjectA.transform.rotation);
        reverseObjectB.name = "reverseObjectB";
        reverseObjectB.GetComponent<Renderer>().material.color = Color.blue;
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
            float targetRelativePositionAngle = 90;
            reverseObjectA.transform.rotation = refObjectA.transform.rotation;
            reverseObjectA.transform.position = Quaternion.AngleAxis(
                targetRelativePositionAngle,
                Vector3.Cross(refObjectA.transform.rotation * Vector3.up, -(refObjectA.transform.position - reverseObjectA.transform.position)).normalized
            ) * (reverseObjectA.transform.rotation * Vector3.up) + refObjectA.transform.position;
            //reverseObjectA.transform.position = refObjectA.transform.position + Vector3.forward;

            reverseObjectB.transform.rotation = refObjectB.transform.rotation;
            reverseObjectB.transform.position = Quaternion.AngleAxis(
                targetRelativePositionAngle,
                Vector3.Cross(refObjectB.transform.rotation * Vector3.up, -(refObjectB.transform.position - reverseObjectB.transform.position).normalized)
            ) * (reverseObjectB.transform.rotation * Vector3.up) + refObjectB.transform.position;
            //reverseObjectB.transform.position = Vector3.Cross(refObjectA.transform.rotation * Vector3.up, -(refObjectA.transform.position - reverseObjectA.transform.position).normalized) + refObjectA.transform.position;
            //Debug.Log("Cross " + Vector3.Cross(refObjectB.transform.rotation * Vector3.up, -(refObjectB.transform.position - reverseObjectA.transform.position).normalized).ToString("F4"));
        }
    }
}
