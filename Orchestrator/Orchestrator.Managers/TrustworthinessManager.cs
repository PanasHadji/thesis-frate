using System.Text;
using Elsa.Workflows;

namespace Orchestrator.Managers;

public class TrustworthinessManager
{
    public async Task CalculateByRules(ActivityExecutionContext activityContext, IHttpClientFactory httpClientFactory,
        string jsonString, string stagingFolderName)
    {
        HttpClient httpClient = httpClientFactory.CreateClient(nameof(TrustworthinessManager));

        HttpRequestMessage request = new HttpRequestMessage(HttpMethod.Post, "http://calculate-trust-stats:8080/");
        
        StringContent content = new StringContent(jsonString, Encoding.UTF8, "application/json");
        request.Content = content;

        CancellationToken cancellationToken = activityContext.CancellationToken;
        HttpResponseMessage response = await httpClient.SendAsync(request, cancellationToken);
    }
}