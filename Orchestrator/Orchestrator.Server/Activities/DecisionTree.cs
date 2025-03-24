using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Elsa.Workflows.UIHints;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Infrastructure.Enums.DecisionTree;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "SLICES", DisplayName = "Decision Tree", Description = "Decision Tree Classifier used using sklearn")]
public class DecisionTree : PythonFaasActivityBase
{
    #region Inputs
    [Input(Description = "The Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe { get; set; } = default!;
    
    [Input(
        Description = "Splitting Criterion",
        UIHint = InputUIHints.DropDown,
        Options = new[] { nameof(CriterionEnum.Gini), nameof(CriterionEnum.Entropy) } ,
        DefaultValue = CriterionEnum.Gini,
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Criterion { get; set; } = default!;
    
    [Input(
        Description = "Splitter",
        UIHint = InputUIHints.DropDown,
        Options = new[] { nameof(SplitterEnum.Best), nameof(SplitterEnum.Random) } ,
        DefaultValue = SplitterEnum.Best,
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Splitter { get; set; } = default!;

    [Input(DisplayName = "Maximum Depth", Description = "The maximum depth of the tree")]
    public Input<int>? MaxDepth { get; set; } = default!;

    [Input(DisplayName = "Minimum Samples Split", Description = "The minimum number of samples required to split an internal node")]
    public Input<float>? MinSamplesSplit { get; set; } = default!;    
    
    [Input(DisplayName = "Minimum Samples Leaf", Description = "The minimum number of samples required to be at a leaf node")]
    public Input<int>? MinSamplesLeaf { get; set; } = default!;        
    
    [Input(DisplayName = "Minimum Weighted Fraction", Description = "The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node")]
    public Input<float>? MinWeightFraction { get; set; } = default!;   
    
    [Input(DisplayName = "Maximum Features", Description = "The number of features to consider when looking for the best split")]
    public Input<int>? MaxFeatures { get; set; } = default!;   
    
    [Input(DisplayName = "Maximum Leaf Nodes", Description = "")]
    public Input<int>? MaxLeafNodes { get; set; } = default!;    
    
    [Input(DisplayName = "Target Variable", Description = "")]
    public Input<string> TargetVariable { get; set; } = default!; 
    
    [Input(DisplayName = "Test Size", Description = "The proportion (e.g 0.3) of the dataset that will be used to evaluate the model's performance (DEFAULT=0.30)")]
    public Input<float>? TestSize { get; set; } = default!; 
    
    [Output(Description = "The output dataframe.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }
    #endregion

    protected override PythonScriptContext GetPythonContext()
    {
        Console.WriteLine("===> START Decision Tree Node <====");
        
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://decision-tree:8080/", // TODO: send factory client

            Inputs = new Dictionary<string, Input?>{
                ["Dataframe"] = InDataframe,
                ["Criterion"] = Criterion,
                ["Splitter"] = Splitter,
                ["MaxDepth"] = MaxDepth,
                ["MinSamplesSplit"] = MinSamplesSplit,
                ["MinSamplesLeaf"] = MinSamplesLeaf,
                ["MinWeightFraction"] = MinWeightFraction,
                ["MaxFeatures"] = MaxFeatures,
                ["MaxLeafNodes"] = MaxLeafNodes,
                ["TargetVariable"] = TargetVariable,
                ["TestSize"] = TestSize,
            },

            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}