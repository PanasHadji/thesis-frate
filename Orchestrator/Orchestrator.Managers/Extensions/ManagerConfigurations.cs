using Microsoft.Extensions.DependencyInjection;

namespace Orchestrator.Managers.Extensions;

public static class ManagerConfigurations
{
    public static IServiceCollection AddManagerServices(this IServiceCollection services)
    {
        services.AddScoped<TrustworthinessManager>();// TODO: Maybe get D.I interface.

        return services;
    }
    
}