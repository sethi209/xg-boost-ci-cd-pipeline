#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import pandas as pd
import xgboost
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from google.cloud import storage
import os
# import logging
# from pythonjsonlogger import jsonlogger

# Set up logging configuration
# def setup_logger():
#     logger = logging.getLogger("XGBoost_Training")
#     logger.setLevel(logging.INFO)
#     logHandler = logging.StreamHandler()  # Logging to stdout
#     formatter = jsonlogger.JsonFormatter()
#     logHandler.setFormatter(formatter)
#     logger.addHandler(logHandler)
#     return logger

# Upload file to Google Cloud Storage
def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    """Uploads a file to the GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f'File saved to: gs://{bucket_name}/{destination_blob_name}')

# Training function
def train_xgboost_model(n_estimators, max_depth, learning_rate, subsample):
    # Load the California housing dataset
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create XGBoost model with hyperparameters
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    
    # Log the MSE
    # logger.info({"mean_squared_error": mse})

    # Save model locally
    local_model_path = f'model_{n_estimators}_{max_depth}_{learning_rate}_{subsample}.bst'
    model.save_model(local_model_path)

    # Save MSE metric to a text file
    mse_file_path = f'mse_{n_estimators}_{max_depth}_{learning_rate}_{subsample}.txt'
    with open(mse_file_path, 'w') as f:
        f.write(f"Mean Squared Error: {mse}")

    # Define unique paths for each trial in GCS
    bucket_name = 'mlops_task_us_central1'  # Replace with your GCS bucket name
    gcs_model_path = f'model_artifacts/trial_{n_estimators}_{max_depth}_{learning_rate}_{subsample}/model.bst'
    gcs_mse_path = f'model_artifacts/trial_{n_estimators}_{max_depth}_{learning_rate}_{subsample}/mse.txt'

    # Upload model and MSE metric to GCS
    upload_to_gcs(local_model_path, bucket_name, gcs_model_path)
    upload_to_gcs(mse_file_path, bucket_name, gcs_mse_path)

    # Optionally, delete the local files after uploading
    os.remove(local_model_path)
    os.remove(mse_file_path)

    # Return the MSE for logging
    return mse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters for tuning
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--subsample', type=float, default=1.0)
    
    args = parser.parse_args()

#     # Set up the logger
#     logger = setup_logger()

    # Run training and log MSE
    mse = train_xgboost_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
    )

    # Print MSE to stdout to confirm successful completion
    print(f"Trial completed with MSE: {mse}")

