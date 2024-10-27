#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.cloud import aiplatform

aiplatform.init(
    project="mlops-task-439307",
    location="us-central1",
    staging_bucket="gs://mlops_task_us_central1"
)


# In[2]:


model = aiplatform.Model.upload(
    display_name="xgboost_model",
    artifact_uri="gs://mlops_task_us_central1/best_model/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
    serving_container_predict_route="/predict",
    serving_container_health_route="/health"
)

# Create an endpoint for deployment
endpoint = aiplatform.Endpoint.create(display_name="xgboost_endpoint")

# Deploy the model to the endpoint
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name="xgboost_model_deployment",
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=1
)


# In[ ]:




