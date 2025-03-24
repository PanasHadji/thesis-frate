using Elsa.Workflows.Models;

namespace Orchestrator.Infrastructure.Contexes;

public class PythonScriptContext
{
    public string ExecuteUrl { get; set; }
    public Dictionary<string, Input?> Inputs { get; set; } = new();
    public Dictionary<string, Output?> Outputs { get; set; } = new();
}
