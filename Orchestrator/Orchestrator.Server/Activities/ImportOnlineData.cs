using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Elsa.Workflows.UIHints;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Infrastructure.Enums.ImportTextData;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "FRATE", DisplayName = "Import Online Data")]
public class ImportOnlineData : PythonFaasActivityBase
{
    #region Inputs/Outputs

    [Input(Description = "The URL of the file to read.")]
    public Input<string> FileURL { get; set; } = default!;

    [Input(
        Description = "The mode of the file to read (Delimited or Fixed).",
        UIHint = InputUIHints.DropDown,
        Options = new[]
        {
            nameof(ImportFileMode.Delimited),
            nameof(ImportFileMode.Fixed)
        },
        DefaultValue = nameof(ImportFileMode.Delimited),
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Mode { get; set; } = default!;

    [Input(Description = "The delimiter to use for delimited mode.",
        UIHint = InputUIHints.DropDown,
        Options = new[]
        {
            nameof(ImportFileDelimiter.Comma),
            nameof(ImportFileDelimiter.Pipe),
            nameof(ImportFileDelimiter.Semicolon),
            nameof(ImportFileDelimiter.Tab)
        },
        DefaultValue = nameof(ImportFileDelimiter.Comma),
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Delimiter { get; set; }

    [Input(Description = "The line number to start reading from (0-based).")]
    public Input<int>? StartLine { get; set; }

    [Input(Description = "The line number to stop reading (exclusive, 0-based).")]
    public Input<int>? EndLine { get; set; }

    [Output(Description = "The Pickled Dataframe to export.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }

    #endregion

    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://import-online-data:8080/",

            Inputs = new Dictionary<string, Input?>
            {
                ["FileURL"] = FileURL,
                ["Mode"] = Mode,
                ["Delimiter"] = Delimiter,
                ["StartLine"] = StartLine,
                ["EndLine"] = EndLine,
            },

            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
            }
        };
    }
}