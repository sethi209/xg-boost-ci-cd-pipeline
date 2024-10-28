#!/usr/bin/env python
# coding: utf-8

from google.cloud import aiplatform

PROJECT_ID = "mlops-task-439307"
REGION = "us-central1"
BUCKET_NAME = "mlops_task_us_central1"

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket="gs://mlops_task_us_central1")

from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google.cloud.aiplatform import CustomPythonPackageTrainingJob, HyperparameterTuningJob, CustomJob

# Hyperparameter tuning configurations
parameter_spec = {
    "n_estimators": hpt.IntegerParameterSpec(min=50, max=150, scale="linear"),
    "max_depth": hpt.IntegerParameterSpec(min=3, max=10, scale="linear"),
    "learning_rate": hpt.DoubleParameterSpec(min=0.01, max=0.3, scale="linear"),
    "subsample": hpt.DoubleParameterSpec(min=0.5, max=1.0, scale="linear"),
}

metric_spec = {"mean_squared_error": "minimize"}

training_job = CustomPythonPackageTrainingJob(
    display_name="xgboost-hyperparameter-tuning",
    python_package_gcs_uri="gs://mlops_task/my_training_package.zip",
    python_module_name="my_training_package.train",
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-11:latest", 
)

worker_pool_spec = {
    "machine_spec": {
        "machine_type": "n1-standard-4",
    },
    "replica_count": 1,
    "python_package_spec": {
        "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-6:latest",
        "package_uris": ["gs://mlops_task_us_central1/my_training_package-0.2.tar.gz"],
        "python_module": "my_training_package.train",
    },
}

custom_job = CustomJob(
    display_name="xgboost-training-job",
    worker_pool_specs=[worker_pool_spec],
)

tuning_job = HyperparameterTuningJob(
    display_name="xgboost-hyperparam-tuning",
    custom_job=custom_job,
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=2,  # Number of trials
    parallel_trial_count=2,  # Parallel trials
)

# Run the tuning job
tuning_job.run(sync=True)
