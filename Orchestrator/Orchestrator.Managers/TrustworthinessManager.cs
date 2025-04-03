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

        HttpRequestMessage request = new HttpRequestMessage(HttpMethod.Post, "http://calculate-trust-stats:8080/");
        StringContent content = new StringContent(jsonString, Encoding.UTF8, "application/json");
        request.Content = content;

        CancellationToken cancellationToken = activityContext.CancellationToken;

        var retryPolicy = Policy
            .Handle<TaskCanceledException>()
            .Or<HttpRequestException>()
            .WaitAndRetryAsync(3, retryAttempt => TimeSpan.FromSeconds(Math.Pow(2, retryAttempt)));

        try
        {
            HttpResponseMessage response = await retryPolicy.ExecuteAsync(async () =>
            {
                return await httpClient.SendAsync(request, cancellationToken);
            });

            Console.WriteLine(response);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error occurred: {ex.Message}");
            throw;
        }
    }
}