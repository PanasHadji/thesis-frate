using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Elsa.Workflows.UIHints;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Infrastructure.Enums.Clustering;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "FRATE", DisplayName = "Clustering",
    Description = "Clustering Classifier using scikit-learn algorithms")]
public class Clustering : PythonFaasActivityBase
{
    #region Inputs/Outputs

    [Input(Description = "The Pickled Dataframe to be used as input for clustering.")]
    public Input<PickleDfRef>? InDataframe { get; set; } = default!;

    [Input(
        Description = "Select the Clustering Algorithm to apply.",
        UIHint = InputUIHints.DropDown,
        Options = new[]
        {
            nameof(ClusteringAlgorithmEnum.KMeans),
            nameof(ClusteringAlgorithmEnum.DBSCAN),
            nameof(ClusteringAlgorithmEnum.GaussianMixture),
            nameof(ClusteringAlgorithmEnum.AffinityPropagation),
            nameof(ClusteringAlgorithmEnum.Ward),
            nameof(ClusteringAlgorithmEnum.AGNES),
            nameof(ClusteringAlgorithmEnum.HDBSCAN),
            nameof(ClusteringAlgorithmEnum.OPTICS),
            nameof(ClusteringAlgorithmEnum.BIRCH),
            nameof(ClusteringAlgorithmEnum.SpectralClustering)
        },
        DefaultValue = nameof(ClusteringAlgorithmEnum.KMeans),
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Algorithm { get; set; } = default!;

    [Input(DisplayName = "Number of Clusters",
        Description = "The number of clusters to form. (KMeans, Gaussian Mixture, Ward, Spectral Clustering)")]
    public Input<int>? NClusters { get; set; } = default!;

    [Input(DisplayName = "Random State",
        Description =
            "Random seed to ensure reproducibility. (KMeans, Gaussian Mixture, Spectral Clustering)")]
    public Input<int>? RandomState { get; set; } = default!;

    [Input(DisplayName = "Damping",
        Description = "The damping factor between 0.5 and 1.0, used to prevent oscillations. (Affinity Propagation)")]
    public Input<float>? Damping { get; set; } = default!;

    [Input(DisplayName = "Preference",
        Description =
            "Preference parameter, which controls the number of exemplars. More negative values result in fewer clusters. (Affinity Propagation)")]
    public Input<int>? Preference { get; set; } = default!;

    [Input(DisplayName = "Eigen Solver",
        Description =
            "The algorithm to use for computing the eigenvectors. Common values are 'arpack' or 'lobpcg'. (Spectral Clustering)")]
    public Input<string>? EigenSolver { get; set; } = default!;

    [Input(DisplayName = "Affinity",
        Description =
            "Method to compute the affinity matrix. Common values are 'nearest_neighbors' or 'rbf'. (Spectral Clustering)")]
    public Input<string>? Affinity { get; set; } = default!;

    [Input(DisplayName = "Epsilon",
        Description =
            "Maximum distance between two samples to consider them as in the same neighborhood. (DBSCAN, OPTICS)")]
    public Input<float>? Eps { get; set; } = default!;

    [Input(DisplayName = "Minimum Samples",
        Description =
            "The number of samples in a neighborhood to consider a point as a core point.(DBSCAN, OPTICS, HDBSCAN)")]
    public Input<int>? MinSamples { get; set; } = default!;

    [Input(DisplayName = "Maximum Epsilon",
        Description = "The maximum distance for two samples to be considered as neighbors. (OPTICS)")]
    public Input<float>? MaxEps { get; set; } = default!;

    [Input(DisplayName = "Minimum Cluster Size", Description = "The minimum size of clusters. (HDBSCAN)")]
    public Input<int>? MinClusterSize { get; set; } = default!;

    [Input(DisplayName = "Threshold", Description = "The radius of subclusters to merge. (BIRCH)")]
    public Input<float>? Threshold { get; set; } = default!;

    [Input(DisplayName = "Linkage",
        Description =
            "Linkage criterion. Options are 'ward', 'complete', 'average', and 'single'. (AGNES)")]
    public Input<string>? Linkage { get; set; } = default!;

    [Input(DisplayName = "Branching Factor", Description = "Maximum number of subclusters a node can have. (BIRCH)")]
    public Input<int>? BranchingFactor { get; set; } = default!;

    [Output(Description = "The resulting DataFrame with cluster labels.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }

    #endregion

    protected override PythonScriptContext GetPythonContext()
    {
        Console.WriteLine("===> START Clustering Node <====");

        return new PythonScriptContext()
        {
            ExecuteUrl = "http://clustering-v2:8080/",

            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe"] = InDataframe,
                ["Algorithm"] = Algorithm,
                ["N_Clusters"] = NClusters,
                ["Random_State"] = RandomState,
                ["Damping"] = Damping,
                ["Preference"] = Preference,
                ["Eigen_Solver"] = EigenSolver,
                ["Affinity"] = Affinity,
                ["Eps"] = Eps,
                ["Min_Samples"] = MinSamples,
                ["Max_Eps"] = MaxEps,
                ["Min_Cluster_Size"] = MinClusterSize,
                ["Threshold"] = Threshold,
                ["Linkage"] = Linkage,
                ["Branching_Factor"] = BranchingFactor,
            },

            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}