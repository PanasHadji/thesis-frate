# FRaTE: Framework for Trustworthy Experimental Replication

## What is this project about?
FRaTE is an innovative tool designed for researchers to repeat, reproduce, and replicate complex AI (Artificial Intelligence, Machine Learning, and Data Mining) experiments. It ensures the reliability and validity of experimental findings while adhering to trustworthy-by-design principles. 

FRaTE introduces trustworthy indicators to evaluate the operation of experimental components both in isolation and combination using supervised and unsupervised machine learning techniques. Additionally, it features an intuitive user interface (UI) that allows researchers to effortlessly configure and run experimental workflows while ensuring their reproducibility and trustworthiness.

### Aims and Objectives
The specific objectives of the FRaTE project include:
- **O1**: Conducting a systematic review of research works and innovation projects in Trustworthiness Evaluation and Management in Smart Networks and Systems.
- **O2**: Designing and developing flexible Metadata Models to support Trustworthy AI Experimentation based on established standards.
- **O3**: Developing an AI-as-a-Service platform adhering to trustworthy design principles for AI experimentation.
- **O4**: Creating a platform for designing and orchestrating complex AI experiments with built-in trust evaluation and management capabilities.
- **O5**: Testing, validating, and experimentally evaluating the efficiency and effectiveness of the platform.

---

## Prerequisites
To set up FRaTE, ensure the following tools are installed:

### Required Software
- **Docker Desktop** (installed and running) using the WSL2 backend. If Docker Desktop was installed recently, WSL2 is the default backend.
- **Scoop** (Windows package manager)

### Required Tools
Run the following commands to install necessary dependencies:
```sh
scoop bucket add tilt-dev https://github.com/tilt-dev/scoop-bucket
scoop install tilt pack
```

---

## Keycloak Setup
FRaTE requires Keycloak for authentication and identity management. Follow these steps to set it up:

### Add Terraform Variables
1. Navigate to:
   ```sh
   thesis-frate/idp-setup/stef-clients-dev
   ```
2. Create a new file named `terraform.tfvars`.
3. Add the following contents:
   ```hcl
   kc_url = "http://localhost:8080"
   kc_realm_id = "frate"
   kc_client_id = "admin-cli"
   kc_username  = "admin"
   kc_password  = "admin"
   ```

### Run Keycloak Container
1. Open Docker and run the following command to start the Keycloak container:
   ```sh
   docker run -p 8080:8080 -e KC_BOOTSTRAP_ADMIN_USERNAME=admin -e KC_BOOTSTRAP_ADMIN_PASSWORD=admin quay.io/keycloak/keycloak:26.0.8 start-dev
   ```
2. Access Keycloak at `http://localhost:8080` and create a new realm named **"frate"**.

### Running Terraform Commands
1. Navigate to:
   ```sh
   thesis-frate/idp-setup/stef-clients-dev
   ```

2. Install Terraform:
   ```sh
   scoop install terraform
   ```

3. Initialize Terraform:
   ```sh
   terraform init
   ```
4. Apply Terraform changes:
   ```sh
   terraform apply
   ```
5. Type `yes` when prompted.

### Create a Realm User
1. Open Keycloak and navigate to the **frate** realm.
2. Create a new user with the required credentials.

---

## Running the Stack
### Using Tilt:
1. Navigate to the root of the repository.
2. Open a command line and run:
   ```sh
   tilt up
   ```
3. Press the **space bar** to open the Tilt UI.

### Important Notes:
- **Docker** must be running.
- **Keycloak container** needs to be started separately (not included in the stack).
- Tilt will automatically build and start all necessary components (**orchestrator, MinIO, Python nodes**).
- If an individual container build fails, click **"trigger update"** in the Tilt UI to retry.

---

## How to Use  

Follow these steps to use FRaTE effectively:  

### 1. Accessing the Orchestrator Designer  
- After running `tilt up`, press the **space bar** to open the **Tilt UI**.  
- Once all services are running, click on **"orchestrator-designer"**.  
- Click on its **open port** (e.g., `localhost:24006`) to access the designer page.  
![image](https://github.com/user-attachments/assets/7603c69a-ad03-43a0-80ba-3ed1b323a29c)

### 2. Logging in  
- Wait for **Keycloak authentication** to redirect you to the login page.  
- Enter your credentials and log in.
![image](https://github.com/user-attachments/assets/c0597702-24fd-44b6-8f7a-e1a28ed50567)

### 3. Workflow Designer (ELSA Workflow Designer)  
- Open the **menu on the left** to:  
  - **Create new workflows**  
  - **View workflow executions**  
![image](https://github.com/user-attachments/assets/e42c6b53-5907-4951-9138-1ed5443ad4dd)

### 4. Workflow Builder  
- The **Workflow Builder** provides various activities.  
- Currently, we only use:  
  - **Start Activity** (in the **Flow** category).  
  - Activities in the **Classification** and **FRaTE** categories.  
- Other activities are not customized for this thesis demo.  
![image](https://github.com/user-attachments/assets/60fae9a4-a2f9-4098-b746-cb543f898553)

### 5. MinIO Storage  
- In the **Tilt UI**, click on **MinIO** and then on its **open port**.  
- Login using:  
  - **Username**: `minio_admin`  
  - **Password**: `minio_admin_password`  
- You can view the **storage bucket** where:  
  - Datasets are uploaded.  
  - Workflow outputs are stored.  
![image](https://github.com/user-attachments/assets/ae1df56e-3d9b-4cd9-a1c1-5d11fcc12633)

### 6. Importing Predefined Workflows  
- Click on **"Import Workflow"** and use pre-built workflow examples.
- You can find example workflows here: https://github.com/PanasHadji/thesis-frate/tree/main/example-workflows.
  ![image](https://github.com/user-attachments/assets/4f2ee2ee-3a00-40bb-ab67-4847c9e57492)
- Also find dummy datasets that can be uploaded to Minio here: https://github.com/PanasHadji/thesis-frate/tree/main/example-datasets

## Development Mode (for ELSA)
### Running Elsa API in Development Mode
1. Navigate to:
   ```sh
   Orchestrator.Frate/Orchestrator/Orchestrator.Server
   ```
2. Run the following command:
   ```sh
   dotnet run --urls "https://localhost:24000"
   ```

### Running Elsa (UI) Designer in Development Mode
1. Navigate to:
   ```sh
   Orchestrator.Frate/Orchestrator/Orchestrator.Elsa.Designer
   ```
2. Run the following command:
   ```sh
   dotnet run --urls "https://localhost:5002"
   ```
3. Ensure that the **Backend URL configuration** in `Orchestrator.Elsa.Designer/wwwroot/appsettings.json` matches your Elsa API host.

---

## Making Changes
### Existing Python Nodes
- **Modifying `requirements.txt`**: The associated container will be **automatically rebuilt**.
- **Modifying other files**: Changes will be **instantly updated** in the container.

### Adding New Python Nodes
1. **Duplicate an existing node folder**.
2. **Duplicate the corresponding entry** in `docker-compose.yml`.
3. **Duplicate the corresponding entry** in the `Tiltfile` using `kn_func_node()`.

### Elsa / C# Changes
- Any changes will automatically trigger a **full container rebuild** (this process will be optimized in future updates).

