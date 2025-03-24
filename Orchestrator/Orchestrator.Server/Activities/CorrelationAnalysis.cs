using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Elsa.Workflows.UIHints;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "FRATE", DisplayName = "Correlation Analysis")]
public class CorrelationAnalysis : PythonFaasActivityBase
{
    #region Inputs/Outputs
    [Input(Description = "The Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe { get; set; } = default!;
    
    [Input(
        Description = "Select the Correlation method to apply.",
        UIHint = InputUIHints.DropDown,
        Options = new[] { nameof(CorrelationMethodEnum.Pearson), nameof(CorrelationMethodEnum.Spearman), nameof(CorrelationMethodEnum.Kendall) } ,
        DefaultValue = nameof(CorrelationMethodEnum.Pearson),
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Method { get; set; } = default!;

    [Output(Description = "The Name of the file to export.")]
    public Output<string>? FileName { get; set; }
    #endregion
    
    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://correlation-analysis:8080/",

            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe"] = InDataframe,
                ["Method"] = Method
            },

            Outputs = new Dictionary<string, Output?>
            {
                ["FileName"] = FileName,
            }
        };
    }
}