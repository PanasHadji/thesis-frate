using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "FRATE", DisplayName = "Sort Columns")]
public class SortColumns : PythonFaasActivityBase
{
    #region Inputs/Ouputs
    [Input(Description = "The Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe { get; set; } = default!;
    
    [Input(Description = "The Name of the column to sort with (ASC and DESC) after it. Use comma to add multiple columns. Default is 'Descending'.")]
    public Input<string>? ColumnsToSort { get; set; } = default!;

    [Output(Description = "The Pickled Dataframe to export.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }
    #endregion
    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://sort-columns:8080/",
            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe"] = InDataframe,
                ["ColumnsToSort"] = ColumnsToSort
            },
            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}