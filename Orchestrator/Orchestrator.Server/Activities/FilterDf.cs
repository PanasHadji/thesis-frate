using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Elsa.Workflows.UIHints;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Infrastructure.Enums.FilterDf;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "SLICES", DisplayName = "Filter Df")]
public class FilterDf : PythonFaasActivityBase
{
    #region Inputs/Outputs
    [Input(Description = "The Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe { get; set; } = default!;

    [Input(Description = "The Name of the column to filter. Multiple should be separated by comma.")]
    public Input<string>? ColumnsToFilter { get; set; } = default!;

    [Input(Description = "The Numeric threshold for filtering.")]
    public Input<float>? Threshold { get; set; } = default!;
    
    [Input(Description = "The condition to use for the threshold.",
        UIHint = InputUIHints.DropDown,
        Options = new[]
        {
            nameof(FilterConditionEnum.Equal),
            nameof(FilterConditionEnum.NotEqual),
            nameof(FilterConditionEnum.Larger),
            nameof(FilterConditionEnum.LargerEqual),
            nameof(FilterConditionEnum.Smaller),
            nameof(FilterConditionEnum.SmallerEqual),
        },
        DefaultValue = nameof(FilterConditionEnum.Equal),
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Condition { get; set; }

    [Output(Description = "The Pickled Dataframe to export.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }
    #endregion

    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://filter-df:8080/",

            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe"] = InDataframe,
                ["ColumnsToFilter"] = ColumnsToFilter,
                ["Condition"] = Condition,
                ["Threshold"] = Threshold
            },

            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}