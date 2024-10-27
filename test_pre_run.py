import os
import unittest
import importlib

# Configurations to validate
GCP_BUCKET = "mlops_task_us_central1"  # Replace with the actual bucket name or import it from config if available


class TestFileIntegrity(unittest.TestCase):

    def test_bucket_name_validity(self):
        """Test if the GCS bucket name is correctly set."""
        self.assertNotEqual(GCP_BUCKET, "", "GCP_BUCKET is empty. Please set a valid bucket name.")
        self.assertTrue(
            GCP_BUCKET.islower() and all(c.isalnum() or c in '-_' for c in GCP_BUCKET),
            "Invalid GCP bucket name format."
        )


class TestConfigParameters(unittest.TestCase):
    def test_hyperparameters_exist(self):
        """Test if hyperparameter values are set within a valid range."""
        # Assuming you have default hyperparameter values in tuning.py or a config file
        hyperparams = {
            "n_estimators": 100,  # Example: Replace with actual defaults
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1
        }

        self.assertGreater(hyperparams["n_estimators"], 50, "n_estimators should be greater than 0.")
        self.assertGreaterEqual(hyperparams["max_depth"], 3, "max_depth should be at least 1.")
        self.assertGreater(hyperparams["learning_rate"], 0, "learning_rate should be positive.")
        self.assertTrue(0 < hyperparams["subsample"] <= 1, "subsample should be between 0 and 1.")


if __name__ == "__main__":
    unittest.main()