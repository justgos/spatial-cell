using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MathDebug : MonoBehaviour
{
    public GameObject debugObject;
    private List<GameObject> debugObjects = new List<GameObject>();
    private float lastGenerationTime = 0;

    void Start()
    {
        //
    }

    void Update()
    {
        if(lastGenerationTime + 1.0f < Time.time)
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
    }
}
