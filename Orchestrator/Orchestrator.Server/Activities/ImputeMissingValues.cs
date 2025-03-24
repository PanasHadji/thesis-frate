using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "FRATE", DisplayName = "Impute Missing Values")]
public class ImputeMissingValues : PythonFaasActivityBase
{
    #region Inputs/Outputs
    [Input(Description = "The Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe { get; set; } = default!;

    [Output(Description = "The Pickled Dataframe to export.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }
    #endregion
    
    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://impute-missing-values:8080/",

            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe"] = InDataframe
            },

            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}