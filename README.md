# XGBoost Model Training and Deployment Pipeline on Google Cloud Platform

This repository provides a CI/CD pipeline setup to train, tune, evaluate, and deploy an XGBoost model on Google Cloud Platform (GCP) using Vertex AI. The pipeline is designed to automate hyperparameter tuning, model evaluation, selection of the best-performing model, and deployment to an endpoint on GCP.

# Overview

The pipeline enables automated model training and hyperparameter tuning on GCP’s Vertex AI, with model artifacts and metrics stored in Google Cloud Storage (GCS). Key steps in the pipeline include:

1. Model Training: Train an XGBoost model on the California housing dataset with specified hyperparameters.
2. Hyperparameter Tuning: Optimize hyperparameters using Vertex AI’s Hyperparameter Tuning feature.
3. Model Evaluation: Evaluate trained models and select the one with the lowest Mean Squared Error (MSE).
4. Model Deployment: Deploy the selected model to a Vertex AI endpoint.

# Files Description

# train.py

This script trains an XGBoost model on the California housing dataset. The training process includes:

1. Argument Parsing: Hyperparameters (n_estimators, max_depth, learning_rate, and subsample) are passed via command-line arguments (in this case, passed by tuning job).
2. Data Loading and Splitting: Loads the dataset, then splits it into training and validation sets.
3. Model Training and Evaluation: Trains an XGBoost model and evaluates it using Mean Squared Error (MSE) on the validation set.
4. Artifact Storage: Saves the trained model and MSE metric as artifacts in GCS.

# tuning.py

This script leverages Vertex AI’s Hyperparameter Tuning capabilities to optimize model performance.

1. Parameter Specifications: Defines a range for n_estimators, max_depth, learning_rate, and subsample.
2. Custom Job Setup: Configures a training job with Vertex AI, specifying machine type, Docker image, and the Python package to be executed.
3. Tuning Job Execution: Runs multiple trials to find the best-performing model configuration.

# evaluation.py

Evaluates trained models to identify the best-performing one based on MSE.

1. Best Model Selection: Scans the model artifacts in GCS, identifying the model with the lowest MSE.
2. Model Saving: Downloads and re-uploads the best model as model.bst to a dedicated directory in GCS.

# deployment.py

Deploys the best model to a Vertex AI endpoint.

1. Model Upload: Uploads the best model from GCS to Vertex AI.
2. Endpoint Creation and Model Deployment: Creates a Vertex AI endpoint and deploys the model, ready for prediction requests.

# CI/CD Configuration - YAML file

The CI/CD pipeline is defined in .github/workflows/, automating the following steps:

1. Code Checkout and Setup: Checks out the code from GitHub and sets up the Python environment.
2. Pre-run Tests: Runs tests to validate configurations and initial setup.
3. Build Process: Builds the training package and uploads it to GCS.
4. Hyperparameter Tuning and Model Evaluation: Executes tuning.py and evaluation.py to train and select the best model.
5. Model Deployment: Runs deployment.py to deploy the best model.

The CI/CD pipeline is triggered on any push to the main branch for specified files.

# test_pre_run.py

Contains unit tests to validate configurations before the main pipeline execution:

1. Bucket Name Validation: Checks if GCP_BUCKET is valid.
2. Hyperparameter Range Tests: Ensures hyperparameters are within acceptable ranges.

# Getting Started

# Requirements

1. Google Cloud Project: Set up a Google Cloud project and enable Vertex AI, Cloud Storage, and IAM permissions.
2. Service Account Key: A GCP service account key with permissions to Vertex AI and GCS, added to GitHub secrets as GCP_SERVICE_ACCOUNT_KEY.
3. GCP SDK: Install and authenticate the GCP SDK locally for testing.

# Installation

Clone the repository and install dependencies

# Environment Setup

Ensure the following environment variables are configured in your setup:

1. PROJECT_ID: Your Google Cloud project ID.
2. REGION: The GCP region (e.g., us-central1).
3. BUCKET_NAME: Name of your Google Cloud Storage bucket (e.g., mlops_task_us_central1).

# Run Pipeline

You can run the scripts manually to test each step or trigger the pipeline by pushing changes to the main branch on GitHub.

# Requirements

Python 3.9
Vertex AI SDK for Python (google-cloud-aiplatform)
Google Cloud Storage SDK (google-cloud-storage)
XGBoost
Scikit-learn
Pandas

# Acknowledgments

Special thanks to the Google Cloud Platform team for providing Vertex AI, which enables seamless machine learning operations.
