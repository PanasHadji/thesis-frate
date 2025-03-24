using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Elsa.Workflows.UIHints;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Infrastructure.Enums.JoinTables;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;


[Activity("Slices", Category = "SLICES", DisplayName = "Join Tables")]
public class JoinTables : PythonFaasActivityBase
{
    #region Input/Output
    [Input(Description = "The first Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe1 { get; set; } = default!;
    
    [Input(Description = "The second Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe2 { get; set; } = default!;
    
    [Input(Description = "The join method to use.",
        UIHint = InputUIHints.DropDown,
        Options = new[]
        {
            nameof(JoinMethodEnum.INNER),
            nameof(JoinMethodEnum.OUTER),
            nameof(JoinMethodEnum.LEFT),
            nameof(JoinMethodEnum.RIGHT),
        },
        DefaultValue = nameof(JoinMethodEnum.INNER),
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Method { get; set; }
    
    [Input(Description = "The column to join.")]
    public Input<string>? OnColumn { get; set; }

    [Output(Description = "The Pickled Dataframe to export.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }

    #endregion
    
    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://join-tables:8080/",

            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe1"] = InDataframe1,
                ["Dataframe2"] = InDataframe2,
                ["Method"] = Method,
                ["OnColumn"] = OnColumn,
            },

            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}