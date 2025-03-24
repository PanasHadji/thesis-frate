﻿using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "SLICES", DisplayName = "Stats")]
public class Stats : PythonFaasActivityBase
{
    #region Inputs/Outputs
    [Input(Description = "The Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe { get; set; } = default!;

    [Output(Description = "The Name of the file to export.")]
    public Output<string>? FileName { get; set; }
    #endregion
    
    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://stats:8080/",

            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe"] = InDataframe
            },

            Outputs = new Dictionary<string, Output?>
            {
                ["FileName"] = FileName,
            }
        };
    }
}