using Elsa.Studio.Dashboard.Extensions;
using Elsa.Studio.Extensions;
using Elsa.Studio.Login.HttpMessageHandlers;
using Elsa.Studio.Shell.Extensions;
using Elsa.Studio.Workflows.Extensions;
using Elsa.Studio.Workflows.Designer.Extensions;
using Elsa.Studio.Login.Contracts;
using Elsa.Studio.Login.Services;
using Elsa.Studio.Contracts;
using Blazored.LocalStorage;
using Elsa.Studio.Core.BlazorWasm.Extensions;
using Elsa.Studio.Login.BlazorWasm.Extensions;
using Elsa.Studio.Login.BlazorWasm.Services;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Authentication;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using Orchestrator.Elsa.Designer;

// Build the host.
var builder = WebAssemblyHostBuilder.CreateDefault(args);

// Add services for your app here
builder.Services.AddSingleton(new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

// Register the root component for Blazor app (i.e. #app)
builder.RootComponents.Add<App>("#app");

// Register Elsa Studio Shell (i.e. #elsa)
builder.RootComponents.Add<Elsa.Studio.Shell.App>("#elsa");

// Register the component for handling <head> manipulation
builder.RootComponents.Add<HeadOutlet>("head::after");

builder.RootComponents.RegisterCustomElsaStudioElements();

RegisterHttpClient(builder, builder.Services);

builder.Services.AddOidcAuthentication(options =>
{
    options.ProviderOptions.MetadataUrl = "http://localhost:8080/realms/frate/.well-known/openid-configuration";
    options.ProviderOptions.Authority = "http://localhost:8080/realms/frate";
    options.ProviderOptions.ClientId = "frate";
    options.ProviderOptions.ResponseType = "id_token token";
    //options.ProviderOptions.DefaultScopes.Add("Audience");

    options.UserOptions.NameClaim = "preferred_username";
    options.UserOptions.RoleClaim = "roles";
    options.UserOptions.ScopeClaim = "scope";
});

// Register shell services and modules.
builder.Services.AddCore();
builder.Services.AddShell();
builder.Services.AddRemoteBackend(
    elsaClient => elsaClient.AuthenticationHandler = typeof(AuthenticatingApiHttpMessageHandler),
    options => builder.Configuration.GetSection("Backend").Bind(options));

builder.Services.AddBlazoredLocalStorage();
        
// Register JWT services.
builder.Services.AddSingleton<IJwtParser, BlazorWasmJwtParser>();
builder.Services.AddScoped<IJwtAccessor, BlazorWasmJwtAccessor>();

builder.Services.AddDashboardModule();
builder.Services.AddWorkflowsModule();

// Build the application.
var app = builder.Build();

// Run each startup task.
var startupTaskRunner = app.Services.GetRequiredService<IStartupTaskRunner>();
await startupTaskRunner.RunStartupTasksAsync();


// Run the application.
await app.RunAsync();

static void RegisterHttpClient(
    WebAssemblyHostBuilder builder,
    IServiceCollection services)
{
    var httpClientName = "Default";

    services.AddHttpClient(httpClientName,
            client => client.BaseAddress = new Uri(builder.HostEnvironment.BaseAddress))
        .AddHttpMessageHandler<BaseAddressAuthorizationMessageHandler>();

    services.AddScoped(
        sp => sp.GetRequiredService<IHttpClientFactory>()
            .CreateClient(httpClientName));
}