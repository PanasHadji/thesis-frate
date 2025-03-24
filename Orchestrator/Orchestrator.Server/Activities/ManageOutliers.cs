using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Elsa.Workflows.UIHints;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Infrastructure.Enums.ManageOutliers;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "FRATE", DisplayName = "Manage Outliers")]
public class ManageOutliers : PythonFaasActivityBase
{
    #region Inputs/Outputs

    [Input(
        Description = "Outlier Management Method",
        UIHint = InputUIHints.DropDown,
        Options = new[]
        {
            nameof(OutlierMethodEnum.Trim),
            nameof(OutlierMethodEnum.Cap),
            nameof(OutlierMethodEnum.Winsorize)
        },
        DefaultValue = nameof(OutlierMethodEnum.Cap),
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Method { get; set; } = default!;

    [Input(Description = "The Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe { get; set; } = default!;

    [Output(Description = "The Pickled Dataframe to export.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }
    #endregion

    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://manage-outliers:8080/",
            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe"] = InDataframe, 
                ["Method"] = Method
            },
            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}