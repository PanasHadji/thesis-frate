using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "FRATE", DisplayName = "Read CSV")]
public class ReadCsv : PythonFaasActivityBase
{
    #region Inputs/Outputs
    [Input(Description = "The Name of the CSV file to read.")]
    public Input<string> FileName { get; set; } = default!;

    [Output(Description = "The Pickled Dataframe to export.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }
    #endregion
    
    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://read-csv:8080/", 

            Inputs = new Dictionary<string, Input?>
            {
                ["FileName"] = FileName
            },

            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}