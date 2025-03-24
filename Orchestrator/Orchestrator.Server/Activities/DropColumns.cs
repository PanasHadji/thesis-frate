using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "FRATE", DisplayName = "Drop Columns")]
public class DropColumns : PythonFaasActivityBase
{
    #region Inputs/Ouputs
    [Input(Description = "The Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe { get; set; } = default!;
    
    [Input(Description = "The Name of the columns to delete. Use comma to add multiple columns.")]
    public Input<string>? ColumnsToRemove { get; set; } = default!;

    [Output(Description = "The Pickled Dataframe to export.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }
    #endregion
    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://drop-columns:8080/",
            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe"] = InDataframe,
                ["ColumnsToRemove"] = ColumnsToRemove
            },
            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}


