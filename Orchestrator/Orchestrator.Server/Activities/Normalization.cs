using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Elsa.Workflows.UIHints;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Infrastructure.Enums.ManageOutliers;
using Orchestrator.Infrastructure.Enums.Normalization;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "FRATE", DisplayName = "Normalization")]
public class Normalization : PythonFaasActivityBase
{
    #region Inputs/Outputs

    [Input(Description = "The Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe { get; set; } = default!;

    [Input(
        Description = "Outlier Management Method",
        UIHint = InputUIHints.DropDown,
        Options = new[]
        {
            nameof(NormalizationMethodEnum.MinMax),
            nameof(NormalizationMethodEnum.DecimalScaling),
            nameof(NormalizationMethodEnum.Discretization)
        },
        DefaultValue = nameof(NormalizationMethodEnum.MinMax),
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Method { get; set; } = default!;

    [Output(Description = "The Pickled Dataframe to export.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }

    #endregion

    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://normalization:8080/",
            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe"] = InDataframe,
                ["Method"] = Method,
            },
            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}