# STEF

**SLICES Trustworthy Experimental Replication Framework**

# Development process

The instruction below is currently only for Windows, although the project can work on mac and linux as well.

## Prerequisites

You need:

* Docker Desktop (installed and running) using the WSL2 backend
  * The WSL2 backend is default if you installed Docker Desktop recently
* [Scoop](https://scoop.sh/)

## Required tools

Run the following commands:

* `scoop bucket add tilt-dev https://github.com/tilt-dev/scoop-bucket`
* `scoop install tilt pack`

## Running the stack

### Using Tiltup:

In the root of the repository, open command line and run `tilt up`.
Press space bar to bring Tilt UI.

> [!IMPORTANT]  
> Docker needs to be running. And Keycloak container started separately (not included in the stack).

Tilt will automatically build and start all different pieces (orchestrator, minio, python nodes).

> [!NOTE]  
> Sometimes the build process of individual containers fails intermittently.
> Simply click "trigger update" in Tilt UI to restart the process.

### Development mode (only for ELSA):

#### Running Elsa API in development mode:
1) Navigate into `Orchestrator.Stef\Orchestrator\Orchestrator.Server>`
2) Run `dotnet run --urls "https://localhost:24000"`

#### Running Elsa (UI) Designer in development mode:
1) Navigate into `Orchestrator.Stef\Orchestrator\Orchestrator.Elsa.Designer>`
2) Run `dotnet run --urls "https://localhost:5002"`
> Note: Make sure that Backend Url configuration inside `Orchestrator.Elsa.Designer/wwwroot/appsettings.json` matches your Elsa API host.

## Making changes

### Existing python nodes

* `requirements.txt` - the associated container will be automatically rebuilt
* All other files - container will be instantly updated with the new versions of the file

### New python nodes

* Duplicate the folder of an existing node
* Duplicate entry in `docker-compose.yml`
* Duplicate entry in `Tiltfile` (`kn_func_node()`)

### Elsa/C#

Changes will automatically trigger a full container rebuild.
The process is somewhat slow and will be improved at a later stage.

## Keycloak Setup

#### Add Terraform Variables
1) Navigate into `Orchestrator.Stef\idp-setup\stef-clients-dev>`
2) Create a new file named `terraform.tfvars`
3) Add this inside:
```
kc_url = "http://localhost:8080"

kc_realm_id = "stef"

kc_client_id = "admin-cli"
kc_username  = "admin"
kc_password  = "admin"
```

#### Run Keycloak Container
1) Open Docker and run its container `docker run -p 8080:8080 -e KC_BOOTSTRAP_ADMIN_USERNAME=admin -e KC_BOOTSTRAP_ADMIN_PASSWORD=admin quay.io/keycloak/keycloak:26.0.8 start-dev`
2) Create a new realm named "stef"

#### Running Terraform Commands:
1) Navigate into `Orchestrator.Stef\idp-setup\stef-clients-dev>`
2) Run `terraform init`
3) Then run `terraform apply` and type `yes`

#### Create Realm User:
1) Go to stef realm in Keycloak
2) Create a new user with credentails



