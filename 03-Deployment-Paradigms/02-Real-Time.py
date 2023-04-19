# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-5e632009-09f2-491e-89c3-565320d463c8
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Real Time Deployment
# MAGIC 
# MAGIC While real time deployment represents a smaller share of the deployment landscape, many of these deployments represent high value tasks.  This lesson surveys real-time deployment options ranging from proofs of concept to both custom and managed solutions.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Survey the landscape of real-time deployment options
# MAGIC  - Prototype a RESTful service using MLflow
# MAGIC  - Deploy registered models using MLflow Model Serving
# MAGIC  - Query an MLflow Model Serving endpoint for inference using individual records and batch requests
# MAGIC  
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png"> *You need <a href="https://docs.databricks.com/applications/mlflow/model-serving.html#requirements" target="_blank">cluster creation</a> permissions to create a model serving endpoint. The instructor will either demo this notebook or enable cluster creation permission for the students from the Admin console.*

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# DBTITLE 0,--i18n-e8d91a25-55ce-47ee-a353-c53ae793e917
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### The Why and How of Real Time Deployment
# MAGIC 
# MAGIC Real time inference is...<br><br>
# MAGIC 
# MAGIC * Generating predictions for a small number of records with fast results (e.g. results in milliseconds)
# MAGIC * The first question to ask when considering real time deployment is: do I need it?  
# MAGIC   - It represents a minority of machine learning inference use cases &mdash; it's necessary when features are only available at the time of serving
# MAGIC   - Is one of the more complicated ways of deploying models
# MAGIC   - That being said, domains where real time deployment is often needed are often of great business value.  
# MAGIC   
# MAGIC Domains needing real time deployment include...<br><br>
# MAGIC 
# MAGIC  - Financial services (especially with fraud detection)
# MAGIC  - Mobile
# MAGIC  - Ad tech

# COMMAND ----------

# DBTITLE 0,--i18n-76d7a403-2d43-425f-ba56-b45e3c1667a4
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC There are a number of ways of deploying models...<br><br>
# MAGIC 
# MAGIC * Many use REST
# MAGIC * For basic prototypes, MLflow can act as a development deployment server
# MAGIC   - The MLflow implementation is backed by the Python library Flask
# MAGIC   - *This is not intended for production environments*
# MAGIC 
# MAGIC In addition, Databricks offers a managed **MLflow Model Serving** solution. This solution allows you to host machine learning models from Model Registry as REST endpoints that are automatically updated based on the availability of model versions and their stages.
# MAGIC 
# MAGIC For production RESTful deployment, there are two main options...<br><br>
# MAGIC 
# MAGIC * A managed solution 
# MAGIC   - Azure ML
# MAGIC   - SageMaker (AWS)
# MAGIC   - VertexAI (GCP)
# MAGIC * A custom solution  
# MAGIC   - Involve deployments using a range of tools
# MAGIC   - Often using Docker or Kubernetes
# MAGIC * One of the crucial elements of deployment in containerization
# MAGIC   - Software is packaged and isolated with its own application, tools, and libraries
# MAGIC   - Containers are a more lightweight alternative to virtual machines
# MAGIC 
# MAGIC Finally, embedded solutions are another way of deploying machine learning models, such as storing a model on IoT devices for inference.

# COMMAND ----------

# DBTITLE 0,--i18n-94daca86-99a3-4ba1-81d2-7978ef10b940
# MAGIC %md-sandbox
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Prototyping with MLflow
# MAGIC 
# MAGIC MLflow offers <a href="https://www.mlflow.org/docs/latest/models.html#pyfunc-deployment" target="_blank">a Flask-backed deployment server for development purposes only.</a>
# MAGIC 
# MAGIC Let's build a simple model below. This model will always predict 5.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import pandas as pd

class TestModel(mlflow.pyfunc.PythonModel):
  
    def predict(self, context, input_df):
        return 5

model_run_name="pyfunc-model"

with mlflow.start_run() as run:
    model = TestModel()
    mlflow.pyfunc.log_model(artifact_path=model_run_name, python_model=model)
    model_uri = f"runs:/{run.info.run_id}/{model_run_name}"

# COMMAND ----------

# DBTITLE 0,--i18n-f1474008-3f65-4d83-829b-802344f0450b
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC There are a few ways to send requests to the development server for testing purpose:
# MAGIC * using **`click`** library 
# MAGIC * using MLflow Model Serving API
# MAGIC * through CLI using **`mlflow models serve`**
# MAGIC 
# MAGIC In this lesson, we are going to demonstrate how to use both the **`click`** library and MLflow Model Serving API. 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> This is just to demonstrate how a basic development server works. This design pattern (which hosts a server on the driver of your Spark cluster) is not recommended for production.<br>
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Models can be served in this way in other languages as well.

# COMMAND ----------

# DBTITLE 0,--i18n-cbe5b173-2e3c-4297-b3b9-0ce3955d1f75
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Method 1: Using `click` Library

# COMMAND ----------

import time
from multiprocessing import Process

server_port_number = 6501
host_name = "127.0.0.1"

def run_server():
    try:
        import mlflow.models.cli
        from click.testing import CliRunner

        CliRunner().invoke(mlflow.models.cli.commands, 
                         ["serve", 
                          "--model-uri", model_uri, 
                          "-p", server_port_number, 
                          "-w", 4,
                          "--host", host_name, # "127.0.0.1", 
                          "--no-conda"])
    except Exception as e:
        print(e)

p = Process(target=run_server) # Create a background process
p.start()                      # Start the process
time.sleep(5)                  # Give it 5 seconds to startup
print(p)                       # Print its status to ensure it's runnning

# COMMAND ----------

# DBTITLE 0,--i18n-07ba25b5-7a4b-4215-b084-e0e0549da8bb
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Create an input for our REST input.

# COMMAND ----------

import json

input_json0 = {"dataframe_split": {"index":[0], "columns":[0], "data":[[0]]}}
input_json = json.dumps(input_json0)
input_json

# COMMAND ----------

# DBTITLE 0,--i18n-ba22cdd2-2b75-4c5d-9e11-45d89b865dbd
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Perform a POST request against the endpoint.

# COMMAND ----------

import requests
from requests.exceptions import ConnectionError
from time import sleep

headers = {"Content-type": "application/json"}
url = f"http://{host_name}:{server_port_number}/invocations"

try:
    response = requests.post(url=url, headers=headers, data=input_json)
except ConnectionError:
    print("Connection fails on a Run All.  Sleeping and will try again momentarily...")
    sleep(5)
    response = requests.post(url=url, headers=headers, data=input_json)

print(f"Status: {response.status_code}")
print(f"Value:  {response.text}")

# COMMAND ----------

# DBTITLE 0,--i18n-fc8c9431-22b7-4d0d-adae-986db149f3df
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Do the same in bash.

# COMMAND ----------

# MAGIC %sh (echo -n '{"dataframe_split": {"index": [0], "columns": [0], "data": [[0]]}}') | curl -H "Content-Type: application/json" -d @- http://127.0.0.1:6501/invocations

# COMMAND ----------

# DBTITLE 0,--i18n-f2903c45-3615-41c4-97c3-9d402399f04f
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Clean up the background process.

# COMMAND ----------

p.terminate()

# COMMAND ----------

# DBTITLE 0,--i18n-63f1e88a-69d1-4ee4-8ecb-b760a4d11168
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Method 2: MLflow Model Serving
# MAGIC Now, let's use MLflow Model Serving. 
# MAGIC 
# MAGIC Step 1: We first need to register the model in MLflow Model Registry and load the model. At this step, we don't specify the model stage, so that the stage version would be **`None`**. 
# MAGIC 
# MAGIC You can refer to the MLflow documentation <a href="https://www.mlflow.org/docs/latest/model-registry.html#api-workflow" target="_blank">here</a>.

# COMMAND ----------

# DBTITLE 0,--i18n-af365e9e-4c75-43b1-9fe7-2aa1fc8b7343
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Train a model.

# COMMAND ----------

import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

suffix = DA.unique_name("-")
model_name = f"demo-model_{suffix}"

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

input_example = X_train.head(3)
signature = infer_signature(X_train, pd.DataFrame(y_train))

with mlflow.start_run(run_name="RF Model") as run:
    mlflow.sklearn.log_model(rf, 
                             "model", 
                             input_example=input_example, 
                             signature=signature, 
                             registered_model_name=model_name, 
                             extra_pip_requirements=["mlflow==2.*"]
                            )

# COMMAND ----------

# DBTITLE 0,--i18n-eb94b91f-df88-4483-8321-75a319841713
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Step 2: Run Tests Against Registered Model

# COMMAND ----------

time.sleep(10) # to wait for registration to complete

model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# DBTITLE 0,--i18n-ac061551-c8bc-4f4e-b811-6ddb970ab1b6
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Enable model inference via API call:
# MAGIC 
# MAGIC After calling `log_model`, a new version of the model is saved. To provision a serving endpoint, follow the steps below.
# MAGIC 
# MAGIC 1. Click **Serving** in the left sidebar. If you don't see it, switch to the Machine Learning Persona ([AWS](https://docs.databricks.com/workspace/index.html#use-the-sidebar)|[Azure](https://docs.microsoft.com/azure/databricks//workspace/index#use-the-sidebar)).
# MAGIC 2. Enable serving for your model. See the Databricks documentation for details ([AWS](https://docs.databricks.com/machine-learning/model-inference/serverless/create-manage-serverless-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-inference/serverless/create-manage-serverless-endpoints)).
# MAGIC 
# MAGIC The code below automatically creates a model serving endpoint for you.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

model_serving_endpoint_name = "endpoint-" + model_name 
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None) # provide both a token for the API, which can be obtained from the notebook.

# With the token, we can create our authorization header for our subsequent REST calls
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

instance = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"

def get_latest_model_version(model_name: str):
    client = MlflowClient()
    models = client.get_latest_versions(model_name, stages=["None"])
    for m in models:
        new_model_version = m.version
    return new_model_version

# COMMAND ----------

# DBTITLE 0,--i18n-5e89ca75-0717-49f9-83fa-e0b1f18b89f7
# MAGIC %md
# MAGIC 
# MAGIC Set up configurations for Serverless model serving endpoint:
# MAGIC * Create the serving endpooint if it does not exist yet
# MAGIC * Or update the configuration of the model serving endpoint if it already exists

# COMMAND ----------

my_json = {
    "name": model_serving_endpoint_name,
    "config": {
        "served_models": [{"model_name": model_name,
                           "model_version": get_latest_model_version(model_name=model_name),
                           "workload_size": "Small",
                           "scale_to_zero_enabled": True 
                          }]
    }
}

def create_endpoint(model_serving_endpoint_name):
    # get endpoint status
    url = f"{endpoint_url}/{model_serving_endpoint_name}"
    r = requests.get(url, headers=headers)
    if "RESOURCE_DOES_NOT_EXIST" in r.text:  
        print("Creating this new endpoint: ", f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations")
        re = requests.post(endpoint_url, headers=headers, json=my_json)
   
    else:
        new_model_version = (my_json['config'])['served_models'][0]['model_version']
        print(f"This endpoint existed previously! We are updating it to a new config with a new model version: {new_model_version}")
        # update config
        url = f"{endpoint_url}/{model_serving_endpoint_name}"
        re = requests.put(f"{url}/config", headers=headers, json=my_json['config']) 
        # wait until new config file in place
        retry = True
        total_wait = 0
        while retry:
            r = requests.get(url, headers=headers)
            assert r.status_code == 200, f"Expected an HTTP 200 response when accessing endpoint info, received {r.status_code}"
            endpoint = json.loads(r.text)
            if "pending_config" in endpoint.keys():
                seconds = 10
                print("New config still pending")
                if total_wait < 6000:
                    #if less than 10 mins, keep waiting
                    print(f"Wait for {seconds} seconds")
                    print(f"Total waiting time so far: {total_wait} seconds")
                    time.sleep(seconds)
                    total_wait += seconds
                else:
                    print(f"Stopping,  waited for {total_wait} seconds")
                    retry = False  
            else:
                print("New config in place now!")
                retry = False
    assert re.status_code == 200, f"Expected an HTTP 200 response, received {re.status_code}"

create_endpoint(model_serving_endpoint_name)

# COMMAND ----------

# DBTITLE 0,--i18n-5890c1e6-9b36-4284-9ac2-3945c1c19886
# MAGIC %md
# MAGIC 
# MAGIC It may take a couple of minutes for the endpoint and model to be ready.
# MAGIC 
# MAGIC Define a **wait_for_endpoint()**  function.

# COMMAND ----------

def wait_for_endpoint():
    url =  f"{endpoint_url}/{model_serving_endpoint_name}" 
    while True:
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        status = response.json().get("state", {}).get("ready", {})
       
        if status == "READY": 
            print(status, "-"*80)
            return
        else: 
            print(f"Endpoint not ready ({status}), waiting 10 seconds")
            time.sleep(10) # Wait 10 seconds

# COMMAND ----------

wait_for_endpoint()

# Give the system just a couple extra seconds to transition
time.sleep(5)

# COMMAND ----------

# DBTITLE 0,--i18n-b388aa18-533a-47e0-8a60-0c972c65b421
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Define the **`score_model()`** function.

# COMMAND ----------

def score_model(dataset: pd.DataFrame, timeout_sec=300):
    start = int(time.time())
    print(f"Scoring {model_name}")
    data_json = {"dataframe_split": dataset.to_dict(orient="split")} 
    url =  f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"
  
    while True:
        response = requests.request(method="POST", headers=headers, url=url, json=data_json)
        elapsed = int(time.time()) - start

        if response.status_code == 200: 
            return response.json()
        elif elapsed > timeout_sec: 
            raise Exception(f"Endpoint was not ready after {timeout_sec} seconds")
        elif response.status_code == 503: 
            print("Temporarily unavailable, retry in 5")
            time.sleep(5)
        else: 
            raise Exception(f"Request failed with status {response.status_code}, {response.text}")

# COMMAND ----------

# DBTITLE 0,--i18n-a2b7fc12-fd0z-493f-be4f-793d0a61695b
# MAGIC %md
# MAGIC 
# MAGIC Now that the model serving cluster is in the **`ready`** state, you can send requests to the REST endpoint.

# COMMAND ----------

score_model(X_test)

# COMMAND ----------

# DBTITLE 0,--i18n-10a3d7cc-2659-444b-a75c-eff0298fbb7c
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC You can also optionally transition the model to the **`Staging`** or **`Production`** stage, using <a href="https://www.mlflow.org/docs/latest/model-registry.html#transitioning-an-mlflow-models-stage" target="_blank">MLflow Model Registry</a>. 
# MAGIC 
# MAGIC Sample code is below:
# MAGIC ```
# MAGIC client.transition_model_version_stage(
# MAGIC     name=model_name,
# MAGIC     version=model_version,
# MAGIC     stage="Staging"
# MAGIC )
# MAGIC ```

# COMMAND ----------

# DBTITLE 0,--i18n-2cfbc2cf-737c-47fa-af47-8a4144e5a9ba
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC Once your serving endpoint is ready, **`score_model`** should give you the model inference result. 
# MAGIC 
# MAGIC **OPTIONAL**: To visualize the UI for model serving or to manually create a model serving endpoint, click the **Serving** tab on the navbar. 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/model_serving_screenshot2_1.png" alt="step12" width="1500"/>
# MAGIC 
# MAGIC <br>
# MAGIC <p>Then select the model to enable model serving or to check the status of model serving.</p>
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/model_serving_screenshot2_2.png" alt="step12" width="1500"/>
# MAGIC  <br>
# MAGIC  <p>Use "Query endpoint" to manually input the payload_json and check the inference result.</p>
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/model_serving_screenshot2_3.png" alt="step12" width="1500"/>

# COMMAND ----------

# DBTITLE 0,--i18n-866cc57f-6fad-4198-8836-02b814eec32e
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_warn_24.png"/> **Please be sure to delete any infrastructure you build after the course so you don't incur unexpected expenses.**

# COMMAND ----------

def delete_model_serving_endpoint(model_serving_endpoint_name):
    url =  f"{endpoint_url}/{model_serving_endpoint_name}" 
    response = requests.delete(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    else:
        print(model_serving_endpoint_name, "endpoint is deleted!")

delete_model_serving_endpoint(model_serving_endpoint_name)

# COMMAND ----------

# DBTITLE 0,--i18n-dcd4faa4-aecc-4c49-858c-e9ef037a8758
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## AWS SageMaker
# MAGIC 
# MAGIC - <a href="https://docs.aws.amazon.com/sagemaker/index.html" target="_blank">mlflow.sagemaker</a> can deploy a trained model to SageMaker using a single function: <a href="https://www.mlflow.org/docs/latest/python_api/mlflow.sagemaker.html#mlflow.sagemaker.deploy" target="_blank">**`mlflow.sagemaker.deploy`**</a>
# MAGIC - During deployment, MLflow will use a specialized Docker container with the resources required to load and serve the model. This container is named **`mlflow-pyfunc`**.
# MAGIC - By default, MLflow will search for this container within your AWS Elastic Container Registry (ECR). You can build and upload this container to ECR using the
# MAGIC **`mlflow sagemaker build-and-push-container`** function in MLflow's <a href="https://www.mlflow.org/docs/latest/cli.html#mlflow-sagemaker-build-and-push-container" target="_blank">CLI</a>.  Alternatively, you can specify an alternative URL for this container by setting an environment variable as follows:
# MAGIC 
# MAGIC ```
# MAGIC   # the ECR URL should look like:
# MAGIC   {account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}
# MAGIC   
# MAGIC   image_ecr_url = "<ecr-url>"
# MAGIC   # Set the environment variable based on the URL
# MAGIC   os.environ["SAGEMAKER_DEPLOY_IMG_URL"] = image_ecr_url
# MAGIC   
# MAGIC   # Deploy the model
# MAGIC   mlflow.sagemaker.deploy(
# MAGIC      app_name, # application/model name
# MAGIC      model_uri, # model URI in Model Registry
# MAGIC      image_url=image_ecr_url, region_name=region, mode="create")
# MAGIC   )
# MAGIC ```
# MAGIC - Running deployment and inference in a Databricks notebook requires the Databricks cluster to be configured with an AWS IAM role with permissions to perform these operations.
# MAGIC - Once the endpoint is up and running, the **`sagemaker-runtime`** API in **`boto3`** can query against the REST API:
# MAGIC ```python
# MAGIC client = boto3.session.Session().client("sagemaker-runtime", "{region}")
# MAGIC   
# MAGIC   response = client.invoke_endpoint(
# MAGIC       EndpointName=app_name,
# MAGIC       Body=inputs,
# MAGIC       ContentType='application/json; format=pandas-split'
# MAGIC   )
# MAGIC   preds = response['Body'].read().decode("ascii")
# MAGIC   preds = json.loads(preds)
# MAGIC   print(f"Received response: {preds}")
# MAGIC   ```
# MAGIC 
# MAGIC **Tip**: Each Sagemaker endpoint is scoped to a single region. If deployment is required across regions, Sagemaker endpoints must exist in each region.

# COMMAND ----------

# DBTITLE 0,--i18n-504bb995-6f89-43cd-8318-fdb3c5eb920c
# MAGIC %md-sandbox
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Azure
# MAGIC - AzureML and MLflow can deploy models as <a href="https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models" target="_blank">REST endpoints</a> by using either:
# MAGIC   - **Azure Container Instances**: when deploying through ACI, it automatically registers the model, creates and registers the container (if one doesn't already exist), builds the image, and sets up the endpoint. The endpoint can then be monitored via the AzureML studio UI. **Note that Azure Kubernetes Service is generally recommended for production over ACI.**
# MAGIC   
# MAGIC   <img src="http://files.training.databricks.com/images/mlflow/rest_serving.png" style="height: 700px; margin: 10px"/>
# MAGIC   - **Azure Kubernetes Service**: when deploying through AKS, the K8s cluster is configured as the compute target, use the `deployment_configuration()` <a href="https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice.aks.akswebservice?view=azure-ml-py#deploy-configuration-autoscale-enabled-none--autoscale-min-replicas-none--autoscale-max-replicas-none--autoscale-refresh-seconds-none--autoscale-target-utilization-none--collect-model-data-none--auth-enabled-none--cpu-cores-none--memory-gb-none--enable-app-insights-none--scoring-timeout-ms-none--replica-max-concurrent-requests-none--max-request-wait-time-none--num-replicas-none--primary-key-none--secondary-key-none--tags-none--properties-none--description-none--gpu-cores-none--period-seconds-none--initial-delay-seconds-none--timeout-seconds-none--success-threshold-none--failure-threshold-none--namespace-none--token-auth-enabled-none--compute-target-name-none--cpu-cores-limit-none--memory-gb-limit-none-" target="_blank">function</a> create a json configuration file for the compute target, the model is then registered and the cluster is ready for serving. Because Azure Kubernetes services inlcudes features like load balancing, fallover, etc. it's a more robust production serving option. 
# MAGIC   - Azure Machine Learning endpoints (currently in preview)
# MAGIC - Note that when you're deploying your model on Azure, you'll need to connect the <a href="https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow" target="_blank">MLflow Tracking URI</a> from your Databricks Workspace to your AzureML workspace. Once the connection has been created, experiments can be tracked across the two. 
# MAGIC 
# MAGIC ** Tip:`azureml-mlflow` will need to be installed on the cluster as it is *not* included in the ML runtime **

# COMMAND ----------

# DBTITLE 0,--i18n-0daf8c28-b9f5-4879-b821-415556648e12
# MAGIC %md-sandbox
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## GCP
# MAGIC 
# MAGIC GCP users can train their models on GCP Databricks workspace, log their trained model to MLFlow Model Registry, then deploy production-ready models to <a href="https://cloud.google.com/vertex-ai" target="_blank">Vertex AI</a> and create model-serving endpoint. You need to set up your GCP service account and install MLflow plugin for Google Cloud (`%pip install google_cloud_mlflow`).
# MAGIC 
# MAGIC ####**To set up GCP service account**:
# MAGIC - Create a GCP project, see intructions <a href="https://cloud.google.com/apis/docs/getting-started" target="_blank">here</a>. You can use the project that the Databricks workspace belongs to.
# MAGIC - Enable Vertex AI and Cloud Build APIs of your GCP project
# MAGIC - Create a service account with the following minimum IAM permissions (see instructions <a href="https://cloud.google.com/iam/docs/creating-managing-service-accounts" target="_blank">here</a> to load Mlflow models from GCS, build containers, and deploy the container into a Vertex AI endpoint:
# MAGIC 
# MAGIC ```
# MAGIC cloudbuild.builds.create
# MAGIC cloudbuild.builds.get
# MAGIC storage.objects.create
# MAGIC storage.buckets.create
# MAGIC storage.buckets.get
# MAGIC aiplatform.endpoints.create
# MAGIC aiplatform.endpoints.deploy
# MAGIC aiplatform.endpoints.get
# MAGIC aiplatform.endpoints.list
# MAGIC aiplatform.endpoints.predict
# MAGIC aiplatform.models.get
# MAGIC aiplatform.models.upload
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC - Create a cluster and attach the service account to your cluster. Compute --> Create Cluster --> (After normal configurations are done) Advanced options --> Google Service Account --> type in your Service Account email --> start cluster
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/gcp_image_2.png" style="height: 700px; margin: 10px"/>
# MAGIC 
# MAGIC 
# MAGIC ####**Create an endpoint of a logged model with the MLflow and GCP python API**
# MAGIC - Install the following libraries in a notebook:
# MAGIC ```
# MAGIC %pip install google_cloud_mlflow
# MAGIC %pip install google-cloud-aiplatform
# MAGIC ```
# MAGIC 
# MAGIC - Deployment
# MAGIC 
# MAGIC ```
# MAGIC import mlflow
# MAGIC from mlflow.deployments import get_deploy_client
# MAGIC 
# MAGIC vtx_client = mlflow.deployments.get_deploy_client("google_cloud") # Instantiate VertexAI client
# MAGIC deploy_name = <enter-your-deploy-name>
# MAGIC model_uri = <enter-your-mlflow-model-uri>
# MAGIC deployment = vtx_client.create_deployment(
# MAGIC     name=deploy_name,
# MAGIC     model_uri=model_uri,
# MAGIC     # config={}   # set deployment configurations, see an example: https://pypi.org/project/google-cloud-mlflow/
# MAGIC     )
# MAGIC ```
# MAGIC The code above will do the heavy lifting depolyment, i.e. export the model from MLflow to Google Storage, imports the model from Google Storage, and generates the image in VertexAI. **It might take 20 mins for the whole deployment to complete.** 
# MAGIC 
# MAGIC **Note:**
# MAGIC - If `destination_image_uri` is not set, then `gcr.io/<your-project-id>/mlflow/<deploy_name>` will be used
# MAGIC - Your service account must have access to that storage location in Cloud Build
# MAGIC 
# MAGIC #### Get predictions from the endpoint
# MAGIC 
# MAGIC - First, retrieve your endpoint:
# MAGIC ```
# MAGIC deployments = vtx_client.list_deployments()
# MAGIC endpt = [d["resource_name"] for d in deployments if d["name"] == deploy_name][0]
# MAGIC ```
# MAGIC 
# MAGIC - Then use `aiplatform` module from `google.cloud` to query the generated endpoint. 
# MAGIC ```
# MAGIC from google.cloud import aiplatform
# MAGIC aiplatform.init()
# MAGIC vtx_endpoint = aiplatform.Endpoint(endpt_resource)
# MAGIC arr = X_test.tolist() ## X_test is an array
# MAGIC pred = vtx_endpoint.predict(instances=arr)
# MAGIC ```

# COMMAND ----------

# DBTITLE 0,--i18n-a2c7fb12-fd0b-493f-be4f-896d0a61695b
# MAGIC %md
# MAGIC ## Online feature store real time model serving
# MAGIC 
# MAGIC You can also work with <a href="https://docs.databricks.com/machine-learning/feature-store/online-feature-stores.html" target="_blank">online feature stores </a> for real-time serving of feature values. With real-time serving, you publish feature tables to a low-latency database and deploy the model to a REST endpoint. Databricks Feature Store also supports automatic feature lookup. Please refer to the example notebooks below:
# MAGIC 
# MAGIC - AWS version (at the end of the page) <a href="https://docs.databricks.com/machine-learning/feature-store/automatic-feature-lookup.html" target="_blank">here</a>   
# MAGIC - Azure version <a href="https://learn.microsoft.com/en-us/azure/databricks/_extras/notebooks/source/machine-learning/feature-store-online-example-cosmosdb.html" target="_blank">here</a>

# COMMAND ----------

# DBTITLE 0,--i18n-a2c7fb12-fd0b-493f-be4f-793d0a61695b
# MAGIC %md
# MAGIC 
# MAGIC ## Classroom Cleanup
# MAGIC 
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson:

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# DBTITLE 0,--i18n-03005f88-5bf3-4876-9446-d00bb1c86793
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Review
# MAGIC **Question:** What are the best tools for real time deployment?  
# MAGIC **Answer:** This depends largely on the desired features.  The main tools to consider are a way to containerize code and either a REST endpoint or an embedded model.  This covers the vast majority of real time deployment options.
# MAGIC 
# MAGIC **Question:** What are the best options for RESTful services?  
# MAGIC **Answer:** The major cloud providers all have their respective deployment options.  In the Azure environment, Azure ML manages deployments using Docker images. This provides a REST endpoint that can be queried by various elements of your infrastructure.
# MAGIC 
# MAGIC **Question:** What factors influence REST deployment latency?  
# MAGIC **Answer:** Response time is a function of a few factors.  Batch predictions should be used when needed since it improves throughput by lowering the overhead of the REST connection.  Geo-location is also an issue, as is server load.  This can be handled by geo-located deployments and load balancing with more resources.

# COMMAND ----------

# DBTITLE 0,--i18n-75a2124d-f7c7-4f5b-9990-90e4d2dcf40e
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Lab<br>
# MAGIC 
# MAGIC Start the labs for this lesson, [Real Time Lab]($./Labs/02-Real-Time-Lab)

# COMMAND ----------

# DBTITLE 0,--i18n-e3aceb7f-9a4e-44cc-83b9-84d363e17df5
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find out more information on MLflow's **`pyfunc`**?  
# MAGIC **A:** Check out <a href="https://www.mlflow.org/docs/latest/models.html#pyfunc-deployment" target="_blank">the MLflow documentation</a>
# MAGIC 
# MAGIC **Q:** Where can I learn more about MLflow Model Serving on Databricks?   
# MAGIC **A:** Check out <a href="https://docs.databricks.com/applications/mlflow/model-serving.html#language-python" target="_blank">this MLflow Model Serving on Databricks documentation</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
