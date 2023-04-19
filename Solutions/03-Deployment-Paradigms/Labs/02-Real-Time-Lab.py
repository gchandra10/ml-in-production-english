# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-54771b4e-fe73-4edb-8d87-9d9d4c2d7170
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Lab: Deploying a Real-time Model with MLflow Model Serving
# MAGIC MLflow Model Serving offers a fast way of serving pre-calculated predictions or creating predictions in real time. In this lab, you'll deploy a model using MLflow Model Serving.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Enable MLflow Model Serving for your registered model
# MAGIC  - Compute predictions in real time for your registered model via a REST API request
# MAGIC  
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png"> *You need <a href="https://docs.databricks.com/applications/mlflow/model-serving.html#requirements" target="_blank">cluster creation</a> permissions to create a model serving endpoint. The instructor will either demo this notebook or enable cluster creation permission for the students from the Admin console.*

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# DBTITLE 0,--i18n-ad24ef8d-031e-435c-a1e0-e64de81b936d
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC To start this off, we will need to load the data, build a model, and register that model.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> We're building a random forest model to predict Airbnb listing prices.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
X = df[["bathrooms", "bedrooms", "number_of_reviews"]]
y = df["price"]

# Start run
with mlflow.start_run(run_name="Random Forest Model") as run:
    # Train model
    n_estimators = 10
    max_depth = 5
    regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    regressor.fit(X, y)
    
    # Evaluate model
    y_pred = regressor.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    
    # Log params and metric
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("rmse", rmse)
    
    # Log model
    mlflow.sklearn.log_model(regressor, "model", extra_pip_requirements=["mlflow==2.*"])
    
# Register model
suffix = DA.unique_name("-")
model_name = f"rfr-model_{suffix}"
model_uri = f"runs:/{run.info.run_id}/model"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
model_version = model_details.version

# COMMAND ----------

# DBTITLE 0,--i18n-ad504f00-8d56-4ab4-aa94-6172107a1934
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Enable MLflow Model Serving for the Registered Model
# MAGIC 
# MAGIC Your first task is to enable Model Serving for the model that was just registered.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Enable serving for your model. See the Databricks documentation for details ([AWS](https://docs.databricks.com/machine-learning/model-inference/serverless/create-manage-serverless-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-inference/serverless/create-manage-serverless-endpoints)).
# MAGIC 
# MAGIC 
# MAGIC To visualize the UI for model serving or to manually create a model serving endpoint, you could click the **"Serving"** tab on the navbar.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/model_serving_screenshot2_1.png" alt="step12" width="1500"/>

# COMMAND ----------

# DBTITLE 0,--i18n-44b586ec-ac4f-4a71-9c27-cc6ac5111ec8
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Compute Real-time Predictions
# MAGIC 
# MAGIC Now that your model is registered, you will query the model with inputs. For simplicity, let's serve model version 1.
# MAGIC 
# MAGIC To do this, you'll first need the appropriate token and url. The code below automatically creates the serving endpoint. You need to set up configs as well.

# COMMAND ----------

model_serving_endpoint_name = "endpoint-lab-" + model_name 
# As the best practice, use secret scope for tokens. 
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None) # provide both a token for the API, which can be obtained from the notebook.
# With the token, we can create our authorization header for our subsequent REST calls
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
  }

instance = spark.conf.get("spark.databricks.workspaceUrl")

my_json = {
    "name": model_serving_endpoint_name,
    "config": {
        "served_models": [{"model_name": model_name,
                           "model_version": "1",
                           "workload_size": "Small",
                           "scale_to_zero_enabled": True
                          }]
    }
}

# COMMAND ----------

# DBTITLE 0,--i18n-cd88b865-b030-48f1-83f5-0f2f872cc757
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Enable the endpoint

# COMMAND ----------

import requests

endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
print("Creating this new endpoint: ", f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations")
re = requests.post(endpoint_url, headers=headers, json=my_json)

assert re.status_code == 200, f"Expected an HTTP 200 response, received {re.status_code}"

# COMMAND ----------

# DBTITLE 0,--i18n-3b7cf885-1789-49ac-be65-f61a9f8752d5
# MAGIC %md
# MAGIC 
# MAGIC We can redefine our wait method to ensure that the resources are ready before moving forward.

# COMMAND ----------

import time

def wait_for_endpoint():
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    while True:
        url =  f"{endpoint_url}/{model_serving_endpoint_name}" 
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

# COMMAND ----------

# DBTITLE 0,--i18n-2e33d989-988d-4673-853f-c7e0e568b3f9
# MAGIC %md
# MAGIC 
# MAGIC Next, create a function that takes a single record as input and returns the predicted value from the endpoint.

# COMMAND ----------

# ANSWER
import requests
    
def score_model(dataset: pd.DataFrame, model_serving_endpoint_name: str, token: str):
    data_json = {"dataframe_split": dataset.to_dict(orient="split")} 
    url =  f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"
    
    response = requests.request(method="POST", headers=headers, url=url, json=data_json)
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()

# COMMAND ----------

# DBTITLE 0,--i18n-ae8b57cb-67fe-4d36-91c2-ce4ec405e38e
# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Now, use that function to score a single row of a Pandas DataFrame.

# COMMAND ----------

# ANSWER

single_row_df = pd.DataFrame([[2, 2, 150]], columns=["bathrooms", "bedrooms", "number_of_reviews"])
score_model(single_row_df, model_serving_endpoint_name, token)

# COMMAND ----------

# DBTITLE 0,--i18n-629fa869-2f79-402d-bb4e-ba6495dfed34
# MAGIC %md
# MAGIC 
# MAGIC ### Notes on request format and API versions
# MAGIC 
# MAGIC The model serving endpoint accepts a JSON object as input.
# MAGIC ```
# MAGIC {
# MAGIC   "index": [0],
# MAGIC   "columns": ["bathrooms", "bedrooms", "number_of_reviews"],
# MAGIC   "data": [[2, 2, 150]]
# MAGIC } 
# MAGIC ```
# MAGIC 
# MAGIC With Databricks Model Serving, the endpoint takes a different body format:
# MAGIC ```
# MAGIC {
# MAGIC   "dataframe_split": [
# MAGIC     { "bathrooms": 2, "bedrooms": 2, "number_of_reviews": 150 }
# MAGIC   ]
# MAGIC }
# MAGIC ```

# COMMAND ----------

# DBTITLE 0,--i18n-TBD
# MAGIC %md
# MAGIC 
# MAGIC Delete the serving endpoint

# COMMAND ----------

def delete_model_serving_endpoint(model_serving_endpoint_name):
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    url =  f"{endpoint_url}/{model_serving_endpoint_name}" 
    response = requests.delete(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    else:
        print(model_serving_endpoint_name, "endpoint is deleted!")
        
delete_model_serving_endpoint(model_serving_endpoint_name)

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

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
