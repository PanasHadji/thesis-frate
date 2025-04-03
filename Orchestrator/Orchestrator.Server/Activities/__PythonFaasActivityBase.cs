using System.Diagnostics;
using System.Text;
using Elsa.Workflows;
using Elsa.Workflows.Memory;
using Elsa.Workflows.Models;
using Microsoft.Extensions.Options;
using Newtonsoft.Json;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Contexes;
using Orchestrator.Infrastructure.Models;
using Orchestrator.Managers;

namespace Orchestrator.Server.Activities;

public abstract class PythonFaasActivityBase : CodeActivity
{
    protected abstract PythonScriptContext GetPythonContext();
    private readonly TrustworthinessManager TrustworthinessManager = new();
    private static Variable<PickleDfRef> PickleDf = new();

    protected override async ValueTask ExecuteAsync(ActivityExecutionContext activityContext)
    {
        // Initialize execution info dictionary to capture metrics
        var executionInfo = new Dictionary<string, object>
        {
            { "ExecInfo", "" }, // Timestamp when execution started
            { "StartTime", DateTime.UtcNow }, // Timestamp when execution started
            { "Success", false }, // Indicates whether the execution was successful
            { "Errors", new List<string>() }, // List of errors encountered during execution
            { "ExecutionTimeMs", 0L }, // Total execution time in milliseconds
            { "ResourceUsage", new Dictionary<string, object>() }, // Tracks CPU, memory, and network usage
            { "RetryCount", 0 }, // Number of retries attempted
            { "StatusCode", 0 }, // HTTP status code of the response
            { "ResponseSizeBytes", 0L }, // Size of the response in bytes
            { "DependencyLatencyMs", 0L }, // Latency of external dependencies
            { "WorkflowStepCompletionRate", 1.0 }, // Percentage of steps completed successfully
        };

        var stopwatch = Stopwatch.StartNew(); // Start measuring execution time

        // Declare variables outside the try block to make them accessible in finally
        PythonScriptContext scriptContext = null;
        string stagingFolderName = null;
        string jsonString = "";
        Dictionary<Argument, string> argumentFileNames = null;
        HttpResponseMessage response = null;
        
        var httpClientFactory = activityContext.GetRequiredService<IHttpClientFactory>();
        var httpClient = httpClientFactory.CreateClient(nameof(PythonFaasActivityBase));

        try
        {
            httpClient.Timeout = TimeSpan.FromMinutes(15);

            PickleDf = new Variable<PickleDfRef>();

            scriptContext = GetPythonContext();

            var request = new HttpRequestMessage(HttpMethod.Post, scriptContext.ExecuteUrl);

            stagingFolderName = string.Join("-", [
                activityContext.WorkflowExecutionContext.Id,
                activityContext.Id,
            ]);

            argumentFileNames = new Dictionary<Argument, string>();

            jsonString = PreparePythonContextJson(activityContext, scriptContext, stagingFolderName, argumentFileNames);

            var content = new StringContent(jsonString, Encoding.UTF8, "application/json");
            request.Content = content;

            var cancellationToken = activityContext.CancellationToken;
            
            var linkedCancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            linkedCancellationTokenSource.CancelAfter(TimeSpan.FromMinutes(30));

            // Measure dependency latency
            var dependencyStopwatch = Stopwatch.StartNew();
            response = await httpClient.SendAsync(request, linkedCancellationTokenSource.Token);
            dependencyStopwatch.Stop();
            executionInfo["DependencyLatencyMs"] = dependencyStopwatch.ElapsedMilliseconds;

            // Capture HTTP status code and response size
            executionInfo["StatusCode"] = (int)response.StatusCode;
            executionInfo["ResponseSizeBytes"] = response.Content.Headers.ContentLength ?? 0;

            Console.WriteLine(response);

            // Mark execution as successful
            executionInfo["Success"] = true;
        }
        catch (Exception ex)
        {
            // Log errors
            ((List<string>)executionInfo["Errors"]).Add(ex.Message);
            executionInfo["Success"] = false;
        }
        finally
        {
            stopwatch.Stop();
            executionInfo["ExecutionTimeMs"] = stopwatch.ElapsedMilliseconds;

            // Gather resource usage metrics (example placeholder)
            executionInfo["ResourceUsage"] = GatherResourceUsage();

            executionInfo["ExecInfo"] = jsonString;

            // Serialize execution info to JSON
            var executionInfoJson = JsonConvert.SerializeObject(executionInfo);

            // Pass the execution info JSON to TrustworthinessManager
            await TrustworthinessManager.CalculateByRules(activityContext, httpClientFactory, executionInfoJson,
                stagingFolderName);

            // Ensure HandleOutcome is called even if an exception occurs
            if (scriptContext != null && stagingFolderName != null && argumentFileNames != null)
            {
                await HandleOutcome(activityContext, scriptContext, stagingFolderName, argumentFileNames,
                    (bool)executionInfo["Success"]);
            }
        }
    }

    private Dictionary<string, object> GatherResourceUsage()
    {
        var resourceUsage = new Dictionary<string, object>();

        try
        {
            using (var process = Process.GetCurrentProcess())
            {
                resourceUsage["CpuUsageMs"] =
                    process.TotalProcessorTime.TotalMilliseconds; // CPU time used by the process
                resourceUsage["MemoryUsageBytes"] = process.WorkingSet64; // Memory usage in bytes
                resourceUsage["PeakMemoryUsageBytes"] = process.PeakWorkingSet64; // Peak memory usage in bytes
                resourceUsage["ThreadsCount"] = process.Threads.Count; // Number of active threads
                resourceUsage["HandleCount"] = process.HandleCount; // Number of handles opened by the process
                resourceUsage["StartTime"] = process.StartTime.ToUniversalTime(); // Process start time in UTC
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error gathering resource usage: {ex.Message}");
        }

        return resourceUsage;
    }


    protected virtual async ValueTask HandleOutcome(ActivityExecutionContext activityContext,
        PythonScriptContext scriptContext, string stagingFolderName, Dictionary<Argument, string> argumentFileNames,
        bool hasSucceeded)
    {
        if (!hasSucceeded)
        {
            throw new Exception("Execution failed");
        }

        foreach (var output in scriptContext.Outputs)
        {
            if (output.Value == null)
            {
                activityContext.Set(output.Value,
                    new PickleDfRef
                    {
                        RelativePath = Path.Combine(stagingFolderName, PickleDf.Value!.ToString()!).Replace("\\", "/")
                    }
                );
                continue;
            }

            if (!output.Value.GetType().GetGenericArguments().SequenceEqual([typeof(PickleDfRef)])) continue;

            activityContext.Set(output.Value,
                new PickleDfRef
                {
                    RelativePath = Path.Combine(stagingFolderName, argumentFileNames[output.Value]).Replace("\\", "/")
                }
            );
        }

        await activityContext.CompleteActivityWithOutcomesAsync("Done");
    }

    public static string PreparePythonContextJson(ActivityExecutionContext activityContext,
        PythonScriptContext scriptContext, string stagingFolderPath, Dictionary<Argument, string> argumentFileNames)
    {
        var config = activityContext.GetRequiredService<IOptionsSnapshot<SliceWorkflowConfiguration>>().Value;

        var jsonObject = new Dictionary<string, object>();

        var inputs = new Dictionary<string, object>();
        foreach (var input in scriptContext.Inputs)
        {
            object value;

            // if (input.Value = null)
            // {
            //     // Look for corresponding output in parent context
            //     var parentOutput = GetParentOutput(activityContext, input.Key);
            //     if (parentOutput != null)
            //     {
            //         value = parentOutput;
            //     }
            //     else
            //     {
            //         // Handle null input case as needed
            //         throw new InvalidOperationException($"Missing input value for {input.Key}");
            //     }
            // }
            // else
            if (input.Value != null)
            {
                value = activityContext.Get(input.Value.MemoryBlockReference())!;
                if (value is PickleDfRef pickleDfRef)
                {
                    inputs.Add(input.Key,
                        new Dictionary<string, object>
                            { { "type", "pickleDf" }, { "value", pickleDfRef.RelativePath! } });
                }
                else
                {
                    inputs.Add(input.Key,
                        new Dictionary<string, object> { { "type", "literalJSON" }, { "value", value } });
                }
            }
            else
            {
                inputs.Add(input.Key, new Dictionary<string, object> { { "type", "literalJSON" }, { "value", "" } });
            }
        }

        jsonObject.Add("inputs", inputs);

        var configs = new Dictionary<string, object>();

        var configObjectAk = new Dictionary<string, string>
            { { "type", "literalJSON" }, { "value", config.AccessKey } };
        configs.Add("access_key", configObjectAk);

        var configObjectSk = new Dictionary<string, string>
            { { "type", "literalJSON" }, { "value", config.SecretKey } };
        configs.Add("secret_key", configObjectSk);

        var configObjectB = new Dictionary<string, string> { { "type", "literalJSON" }, { "value", config.Bucket } };
        configs.Add("bucket_name", configObjectB);

        jsonObject.Add("config", configs);

        var outputs = new Dictionary<string, object>();
        foreach (var output in scriptContext.Outputs)
        {
            var outputObject = new Dictionary<string, string>
            {
                { "type", "pickleDf" },
                {
                    "destination",
                    stagingFolderPath + "/" +
                    GetAndStoreArgumentName("output", output.Key, output.Value, argumentFileNames)
                }
            };
            outputs.Add(output.Key, outputObject);
        }

        jsonObject.Add("outputs", outputs);
        string json = JsonConvert.SerializeObject(jsonObject, Formatting.Indented);

        return json;
    }

    private static string GetAndStoreArgumentName(string prefix, string label, Argument? argument,
        Dictionary<Argument, string> argumentFileNames)
    {
        string name = $"{prefix}-{label}";

        argument ??= new Output();

        PickleDf.Value = name;

        argumentFileNames[argument] = name;

        return name;
    }

    private static object? GetParentOutput(ActivityExecutionContext context, string inputName)
    {
        ActivityExecutionContext? parent = context.ParentActivityExecutionContext;
        if (parent == null)
        {
            return null;
        }

        ActivityOutputRegister outputRegister = parent.WorkflowExecutionContext.GetActivityOutputRegister();

        var parentOutByName = outputRegister
            .FindMany(r => r.ContainerId == parent.Id)
            .ToDictionary(r => r.ActivityId, r => r.Value);

        var matchingValue = parentOutByName.Values
            .LastOrDefault(pv =>
                pv is PickleDfRef dfRef && dfRef.RelativePath != null && dfRef.RelativePath.Contains(inputName));

        return matchingValue is PickleDfRef result ? result.RelativePath : string.Empty;
    }
}