using Elsa.Workflows.Attributes;
using Elsa.Workflows.Models;
using Elsa.Workflows.UIHints;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Infrastructure.Enums.ImportTextData;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

[Activity("Slices", Category = "FRATE", DisplayName = "Export To File")]
public class ExportToFile : PythonFaasActivityBase
{
    #region Inputs/Outputs

    [Input(Description = "The Pickled Dataframe to read.")]
    public Input<PickleDfRef>? InDataframe { get; set; } = default!;

    [Input(
        Description = "The mode of the file to export (Delimited or Fixed).",
        UIHint = InputUIHints.DropDown,
        Options = new[]
        {
            nameof(ExportToFileMode.Delimited),
            nameof(ExportToFileMode.Text)
        },
        DefaultValue = nameof(ExportToFileMode.Delimited),
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Mode { get; set; } = default!;

    [Input(Description = "The delimiter to use for delimited mode.",
        UIHint = InputUIHints.DropDown,
        Options = new[]
        {
            nameof(ExportToFileDelimiter.Comma),
            nameof(ExportToFileDelimiter.Pipe),
            nameof(ExportToFileDelimiter.Semicolon),
            nameof(ExportToFileDelimiter.Tab)
        },
        DefaultValue = nameof(ExportToFileDelimiter.Comma),
        SupportedSyntaxes = new[] { "Default" }
    )]
    public Input<string>? Delimiter { get; set; }

    [Input(Description = "The line number to start exporting from (0-based).")]
    public Input<int>? StartLine { get; set; }

    [Input(Description = "The line number to stop exporting (exclusive, 0-based).")]
    public Input<int>? EndLine { get; set; }

    [Output(Description = "The Pickled Dataframe to export.")]
    public Output<PickleDfRef>? OutDataframe { get; set; }
    
    [Output(Description = "The File Name to export.")]
    public Output<string>? FileName { get; set; }

    #endregion

    protected override PythonScriptContext GetPythonContext()
    {
        return new PythonScriptContext()
        {
            ExecuteUrl = "http://export-to-file:8080/",

            Inputs = new Dictionary<string, Input?>
            {
                ["Dataframe"] = InDataframe,
                ["Mode"] = Mode,
                ["Delimiter"] = Delimiter,
                ["StartLine"] = StartLine,
                ["EndLine"] = EndLine,
            },

            Outputs = new Dictionary<string, Output?>
            {
                ["Dataframe"] = OutDataframe,
                ["FileName"] = FileName,
            }
        };
    }
}