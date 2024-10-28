#!/usr/bin/env python
# coding: utf-8

from google.cloud import storage
import os

# Set up Google Cloud Storage client
client = storage.Client()

# Function to find the trial with the minimum MSE
def find_best_model(bucket_name, base_path):
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=base_path)

    min_mse = float('inf')
    best_model_path = None

    # Loop through blobs to find MSE files
    for blob in blobs:
        if blob.name.endswith('mse.txt'):
            # Download MSE file
            mse_file_path = f'/tmp/{blob.name.split("/")[-1]}'
            blob.download_to_filename(mse_file_path)

            # Read MSE value
            with open(mse_file_path, 'r') as f:
                mse_value = float(f.readline().strip().split(': ')[1])

            # Compare MSE values
            if mse_value < min_mse:
                min_mse = mse_value
                # Derive the model path based on MSE file path
                model_file_path = blob.name.replace('mse.txt', 'model.bst')
                best_model_path = model_file_path

            # Clean up local MSE file
            os.remove(mse_file_path)

    return best_model_path, min_mse

# Function to save the best model as 'model.bst' in GCS
def save_best_model(bucket_name, best_model_path, destination_path='best_model/model.bst'):
    if best_model_path:
        # Download the best model
        bucket = client.bucket(bucket_name)
        model_blob = bucket.blob(best_model_path)
        model_blob.download_to_filename('/tmp/best_model.bst')

        # Upload as 'model.bst'
        destination_blob = bucket.blob(destination_path)
        destination_blob.upload_from_filename('/tmp/best_model.bst')
        print(f'Best model with lowest MSE saved as gs://{bucket_name}/{destination_path}')

        # Clean up local model file
        os.remove('/tmp/best_model.bst')
    else:
        print("No model found.")

# Specify GCS bucket name and base path where trial artifacts are stored
bucket_name = 'mlops_task_us_central1'  
base_path = 'model_artifacts/'  # Directory where trial results are saved

# Find the best model and save it as model.bst in GCS
best_model_path, min_mse = find_best_model(bucket_name, base_path)
print(f"Best MSE: {min_mse} for model at {best_model_path}")
save_best_model(bucket_name, best_model_path)


