using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Elsa.Workflows.UIHints;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Infrastructure.Enums.NearestNeighbors;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "Classification", DisplayName = "Nearest Neighbors")]
public class NearestNeighbors : PythonFaasActivityBase
{
    #region Inputs/Outputs
    [Input(Description = "The Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe { get; set; } = default!;
    
    [Input(Description = "The Number of Neighbors to pick.")]
    public Input<int>? Neighbors { get; set; } = default!;
    
    [Input(
        Description = "Select the Clustering Algorithm to apply.",
        UIHint = InputUIHints.DropDown,
        Options = new[]
        {
            nameof(DistanceMetricEnum.Euclidean),
            nameof(DistanceMetricEnum.Manhattan),
            nameof(DistanceMetricEnum.Minkowski),
            nameof(DistanceMetricEnum.Chebyshev),
            nameof(DistanceMetricEnum.Hamming),
            nameof(DistanceMetricEnum.Cosine),
            nameof(DistanceMetricEnum.Jaccard),
            nameof(DistanceMetricEnum.Mahalanobis),
            nameof(DistanceMetricEnum.Canberra),
            nameof(DistanceMetricEnum.BrayCurtis)
        },
        DefaultValue = nameof(DistanceMetricEnum.Euclidean),
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Metric { get; set; } = default!;
    
    [Output(Description = "The Pickled Dataframe to export.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }
    #endregion
    
    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://nearest-neighbors:8080/",

            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe"] = InDataframe,
                ["N_Neighbors"] = Neighbors,
                ["Metric"] = Metric,
            },

            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}