using UnityEngine;
using System.Linq;
using System.Collections.Generic;
using TMPro; // Important for TextMeshPro

public class KMeansController : MonoBehaviour
{
    // === PUBLIC CONFIGURATION (Set in Unity Inspector) ===
    public int K = 3; // Number of clusters (Start with 3 or 4)
    public int maxIterations = 100;
    public GameObject pointPrefab;      // Assign your Sphere Prefab
    public GameObject textLabelPrefab;  // Assign your CollegeNameLabel Prefab
    public float visualizationScale = 10f;

    // === PRIVATE DATA STRUCTURES ===
    private class CollegeData
    {
        public string Name;
        public Vector2 NormalizedFeatures;
        public int ClusterID;
    }

    private List<CollegeData> dataPoints = new List<CollegeData>();
    private Color[] clusterColors = { Color.red, Color.green, Color.blue, Color.yellow, Color.cyan };

    void Start()
    {
        LoadAndNormalizeData();

        if (dataPoints.Count > 0)
        {
            Vector2[] centroids = RunKMeans(K, maxIterations);
            VisualizeClustering();
        }
        else
        {
            Debug.LogError("Data failed to load. Check CSV formatting and path.");
        }
    }

    // --- Data Loading & Normalization ---
    private void LoadAndNormalizeData()
    {
        TextAsset dataAsset = Resources.Load<TextAsset>("College_Data");
        string[] lines = dataAsset.text.Split('\n');
        List<Vector2> rawVectors = new List<Vector2>();

        const int TOP10_INDEX = 5; // Top10perc column
        const int TOP25_INDEX = 6; // Top25perc column 

        for (int i = 1; i < lines.Length; i++)
        {
            string line = lines[i].Trim();
            if (string.IsNullOrEmpty(line)) continue;

            string[] splitLine = line.Split(';'); // Using semi-colon delimiter

            if (splitLine.Length > TOP25_INDEX)
            {
                if (float.TryParse(splitLine[TOP10_INDEX], out float top10) &&
                    float.TryParse(splitLine[TOP25_INDEX], out float top25))
                {
                    dataPoints.Add(new CollegeData { Name = splitLine[0].Trim('"'), NormalizedFeatures = Vector2.zero, ClusterID = -1 });
                    rawVectors.Add(new Vector2(top10, top25));
                }
            }
        }

        // --- Z-Score Calculation ---
        if (rawVectors.Count == 0) return;
        Vector2 mean = Vector2.zero;
        foreach (Vector2 point in rawVectors) { mean += point; }
        mean /= rawVectors.Count;

        Vector2 stdDev = Vector2.zero;
        foreach (Vector2 point in rawVectors)
        {
            stdDev += new Vector2(Mathf.Pow(point.x - mean.x, 2), Mathf.Pow(point.y - mean.y, 2));
        }
        stdDev = new Vector2(Mathf.Sqrt(stdDev.x / rawVectors.Count), Mathf.Sqrt(stdDev.y / rawVectors.Count));

        // Apply Normalization
        for (int i = 0; i < rawVectors.Count; i++)
        {
            dataPoints[i].NormalizedFeatures = new Vector2(
                (rawVectors[i].x - mean.x) / (stdDev.x == 0 ? 1 : stdDev.x),
                (rawVectors[i].y - mean.y) / (stdDev.y == 0 ? 1 : stdDev.y)
            );
        }
    }

    // --- K-Means Algorithm Core ---
    private Vector2[] RunKMeans(int k, int maxIter)
    {
        // 1. Initialization: Randomly select K initial centroids
        Vector2[] currentCentroids = dataPoints.OrderBy(x => System.Guid.NewGuid()).Take(k).Select(x => x.NormalizedFeatures).ToArray();

        for (int iter = 0; iter < maxIter; iter++)
        {
            bool changed = AssignToClusters(currentCentroids);
            Vector2[] newCentroids = CalculateNewCentroids(k);

            if (!changed || IsConverged(currentCentroids, newCentroids))
            {
                return newCentroids;
            }

            currentCentroids = newCentroids;
        }
        return currentCentroids;
    }

    private float DistanceSquared(Vector2 a, Vector2 b)
    {
        return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    }

    private bool AssignToClusters(Vector2[] centroids)
    {
        bool changed = false;
        for (int i = 0; i < dataPoints.Count; i++)
        {
            float minDistance = float.MaxValue;
            int bestCluster = -1;

            for (int j = 0; j < centroids.Length; j++)
            {
                float dist = DistanceSquared(dataPoints[i].NormalizedFeatures, centroids[j]);
                if (dist < minDistance)
                {
                    minDistance = dist;
                    bestCluster = j;
                }
            }

            if (dataPoints[i].ClusterID != bestCluster)
            {
                dataPoints[i].ClusterID = bestCluster;
                changed = true;
            }
        }
        return changed;
    }

    private Vector2[] CalculateNewCentroids(int k)
    {
        Vector2[] newCentroids = new Vector2[k];
        int[] clusterCounts = new int[k];

        foreach (var point in dataPoints)
        {
            if (point.ClusterID >= 0 && point.ClusterID < k)
            {
                newCentroids[point.ClusterID] += point.NormalizedFeatures;
                clusterCounts[point.ClusterID]++;
            }
        }

        for (int i = 0; i < k; i++)
        {
            if (clusterCounts[i] > 0)
            {
                newCentroids[i] /= clusterCounts[i];
            }
        }
        return newCentroids;
    }

    private bool IsConverged(Vector2[] oldCentroids, Vector2[] newCentroids, float tolerance = 0.0001f)
    {
        for (int i = 0; i < oldCentroids.Length; i++)
        {
            if (DistanceSquared(oldCentroids[i], newCentroids[i]) > tolerance)
            {
                return false;
            }
        }
        return true;
    }

    // --- 3. VISUALIZATION ---
    private void VisualizeClustering()
    {
        foreach (var point in dataPoints)
        {
            if (point.ClusterID != -1 && pointPrefab != null)
            {
                // 1. Instantiate the College Dot (Sphere)
                GameObject sphere = Instantiate(pointPrefab, transform);
                Vector3 position = new Vector3(
                    point.NormalizedFeatures.x * visualizationScale,
                    point.NormalizedFeatures.y * visualizationScale,
                    0
                );
                sphere.transform.position = position;

                // Color the sphere
                Renderer renderer = sphere.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material.color = clusterColors[point.ClusterID % clusterColors.Length];
                }

                sphere.name = $"{point.Name} - Cluster {point.ClusterID}";

                // 2. Instantiate and Position the College Name Label
                if (textLabelPrefab != null)
                {
                    GameObject label = Instantiate(textLabelPrefab, transform);

                    // Position the label slightly offset from the sphere
                    label.transform.position = position + new Vector3(
                        0.5f, // Offset in X (right)
                        0.5f, // Offset in Y (up)
                        0
                    );

                    // Set the text content
                    TMPro.TextMeshPro textComponent = label.GetComponent<TMPro.TextMeshPro>();
                    if (textComponent != null)
                    {
                        textComponent.text = point.Name;
                        textComponent.color = Color.white;
                    }
                }
            }
        }
    }
}