using System.Text;
using Elsa.Workflows;
using Polly;

namespace Orchestrator.Managers;

public class TrustworthinessManager
{
    public async Task CalculateByRules(ActivityExecutionContext activityContext, IHttpClientFactory httpClientFactory,
        string jsonString, string stagingFolderName)
    {
        HttpClient httpClient = httpClientFactory.CreateClient(nameof(TrustworthinessManager));
        httpClient.Timeout = TimeSpan.FromMinutes(15);

        // Retry policy definition
        var retryPolicy = Policy
            .Handle<TaskCanceledException>()
            .Or<HttpRequestException>()
            .WaitAndRetryAsync(3, retryAttempt => TimeSpan.FromSeconds(Math.Pow(2, retryAttempt)));

        CancellationToken cancellationToken = activityContext.CancellationToken;

        try
        {
            // Execute retry logic
            HttpResponseMessage response = await retryPolicy.ExecuteAsync(async () =>
            {
                // Create a new HttpRequestMessage for each retry attempt
                HttpRequestMessage request = new HttpRequestMessage(HttpMethod.Post, "http://calculate-trust-stats:8080/");
                StringContent content = new StringContent(jsonString, Encoding.UTF8, "application/json");
                request.Content = content;

                // Send the request asynchronously with the cancellation token
                return await httpClient.SendAsync(request, cancellationToken);
            });

            // Log the response (or process it accordingly)
            Console.WriteLine(response);
        }
        catch (Exception ex)
        {
            // Log and handle errors
            Console.WriteLine($"Error occurred: {ex.Message}");
            throw;
        }
    }
}