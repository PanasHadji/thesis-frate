## Stage 1: Build the application
#FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
#WORKDIR /src
#
## Copy the project file and restore dependencies
#COPY ["Orchestrator.Elsa.Designer/Orchestrator.Elsa.Designer.csproj", "Orchestrator.Elsa.Designer/"]
#RUN dotnet restore "Orchestrator.Elsa.Designer/Orchestrator.Elsa.Designer.csproj"
#
## Copy the rest of the application code
#COPY . .
#WORKDIR "/src/Orchestrator.Elsa.Designer"
#
## Build and publish the app
#RUN dotnet publish "Orchestrator.Elsa.Designer.csproj" -c Release -o /app/publish
#
## Stage 2: Use a lightweight static file server
#FROM nginx:stable-alpine AS final
#WORKDIR /usr/share/nginx/html
#
## Copy the published Blazor WebAssembly output
#COPY --from=build /app/publish/wwwroot .
#
## Expose the default HTTP port
#EXPOSE 80
#
## Start the Nginx server
#ENTRYPOINT ["nginx", "-g", "daemon off;"]

####################3
# Stage 1: Build the application
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy the project file and restore dependencies
COPY ["Orchestrator.Elsa.Designer/Orchestrator.Elsa.Designer.csproj", "Orchestrator.Elsa.Designer/"]
RUN dotnet restore "Orchestrator.Elsa.Designer/Orchestrator.Elsa.Designer.csproj"

# Copy the rest of the application code
COPY . .
WORKDIR "/src/Orchestrator.Elsa.Designer"

# Build and publish the app
RUN dotnet publish "Orchestrator.Elsa.Designer.csproj" -c Release -o /app/publish

# Stage 2: Use a lightweight static file server
FROM nginx:stable-alpine AS final
WORKDIR /usr/share/nginx/html

# Copy the custom NGINX configuration
COPY Orchestrator.Elsa.Designer/nginx.conf /etc/nginx/conf.d/default.conf

# Copy the published Blazor WebAssembly output
COPY --from=build /app/publish/wwwroot .

# Expose the default HTTP port
EXPOSE 80

# Start the Nginx server
ENTRYPOINT ["nginx", "-g", "daemon off;"]
