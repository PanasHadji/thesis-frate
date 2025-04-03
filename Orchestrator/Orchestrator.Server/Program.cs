using Elsa.EntityFrameworkCore.Extensions;
using Elsa.EntityFrameworkCore.Modules.Management;
using Elsa.EntityFrameworkCore.Modules.Runtime;
using Elsa.Extensions;
using Elsa.Workflows.Activities;
using FastEndpoints.Swagger;
using Keycloak.AuthServices.Authentication;
using Keycloak.AuthServices.Authorization;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Authorization;
using Microsoft.IdentityModel.Tokens;
using Microsoft.OpenApi.Models;
using Orchestrator.Infrastructure;
using Orchestrator.Infrastructure.Models;
using Orchestrator.Managers.Extensions;
using Orchestrator.Server.Activities;
using Prometheus;


var builder = WebApplication.CreateBuilder(args);

// Configure CORS to allow designer app hosted on a different origin to invoke the APIs.
builder.Services.AddCors(cors => cors
    .AddDefaultPolicy(policy => policy
        .AllowAnyOrigin() // For demo purposes only. Use a specific origin instead.
        .AllowAnyHeader()
        .AllowAnyMethod()
        .WithExposedHeaders(
            "x-elsa-workflow-instance-id"))); // Required for Elsa Studio in order to support running workflows from the designer. Alternatively, you can use the `*` wildcard to expose all headers.

// Add services to the container.
Elsa.EndpointSecurityOptions.DisableSecurity();

builder.Services.AddControllers();
var con = builder.Configuration.GetConnectionString("elsadb");

var identitySection = builder.Configuration.GetSection("Identity");
var identityTokenSection = identitySection.GetSection("Tokens");

builder.Services.AddElsa(elsa =>
{
    elsa.UseSasTokens(); // Optional for token-based auth

    elsa.UseWorkflowManagement(management => { management.AddVariableType<PickleDfRef>(category: "FRATE"); });

    elsa.UseIdentity(identity =>
    {
        identity.IdentityOptions = options => identitySection.Bind(options);
        identity.TokenOptions = options => identityTokenSection.Bind(options);
        identity.UseConfigurationBasedUserProvider(options => identitySection.Bind(options));
        identity.UseConfigurationBasedApplicationProvider(options => identitySection.Bind(options));
        identity.UseConfigurationBasedRoleProvider(options => identitySection.Bind(options));
        identity.UseAdminUserProvider();
    });

    var migrationsAssembly = typeof(Elsa.EntityFrameworkCore.PostgreSql.RuntimeDbContextFactory).Assembly;
    var connectionString = builder.Configuration.GetConnectionString("elsadb")!;

    elsa.UseWorkflowManagement(management =>
        management.UseEntityFrameworkCore(ef =>
            ef.DbContextOptionsBuilder = (_, db) => db.UseElsaPostgreSql(migrationsAssembly, connectionString, null,
                configure => configure.CommandTimeout(60000))
        ));
    elsa.UseWorkflowRuntime(runtime =>
        runtime.UseEntityFrameworkCore(ef =>
            ef.DbContextOptionsBuilder = (_, db) => db.UseElsaPostgreSql(migrationsAssembly, connectionString, null,
                configure => configure.CommandTimeout(60000))
        ));
    
    elsa.UseJavaScript();
    elsa.UseLiquid();
    elsa.UseHttp();
    elsa.UseWorkflowsApi();

    // Use timers.
    elsa.UseQuartz();
    elsa.UseScheduling(scheduling => scheduling.UseQuartzScheduler());

    elsa.AddActivitiesFrom<Program>();
    elsa.AddWorkflowsFrom<Program>();
    elsa.AddActivitiesFrom<PythonFaasActivityBase>();
});
builder.Services.AddManagerServices();
builder.Services.AddOptions<SliceWorkflowConfiguration>().BindConfiguration("Slices:Workflow");

builder.Services.AddManagerServices();
builder.Services.AddHealthChecks();

builder.Services.AddEndpointsApiExplorer();
var openIdConnectUrl =
    $"{builder.Configuration["Keycloak:auth-server-url"]}realms/{builder.Configuration["Keycloak:realm"]}/.well-known/openid-configuration";

builder.Services.AddSwaggerGen(c =>
{
    var securityScheme = new OpenApiSecurityScheme
    {
        Name = "Auth",
        In = ParameterLocation.Header,
        Type = SecuritySchemeType.OpenIdConnect,
        OpenIdConnectUrl = new Uri(openIdConnectUrl),
        Scheme = "bearer",
        BearerFormat = "JWT",
        Reference = new OpenApiReference
        {
            Id = "Bearer",
            Type = ReferenceType.SecurityScheme
        }
    };
    c.AddSecurityDefinition(securityScheme.Reference.Id, securityScheme);
    c.AddSecurityRequirement(new OpenApiSecurityRequirement
    {
        { securityScheme, Array.Empty<string>() }
    });
});

// Add Health Checks (optional)
builder.Services.AddHealthChecks();

// Build the web application
var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    SwaggerBuilderExtensions.UseSwagger(app);
    app.UseSwaggerUI();
}

app.UseCors();
app.UseRouting();
app.UseAuthentication();
app.UseAuthorization();
app.UseSwaggerGen();
app.MapControllers();
app.UseWorkflows();
app.UseWorkflowsApi();
app.UseHttpMetrics(); // Enable metrics collection
app.UseEndpoints(endpoints =>
{
    endpoints.MapMetrics(); // Expose metrics at /metrics
});
app.Run();