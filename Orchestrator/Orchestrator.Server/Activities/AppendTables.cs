using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;


[Activity("Slices", Category = "FRATE", DisplayName = "Append Tables")]
public class AppendTables : PythonFaasActivityBase
{
    #region Input/Output
    [Input(Description = "The first Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe1 { get; set; } = default!;
    
    [Input(Description = "The second Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe2 { get; set; } = default!;

    [Output(Description = "The Pickled Dataframe to export.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }

    #endregion
    
    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://append-tables:8080/",

            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe1"] = InDataframe1,
                ["Dataframe2"] = InDataframe2
            },

            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}