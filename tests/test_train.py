# tests/test_train.py

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from src.utils import retrieve_model, fetch_data_split

# Define a minimum acceptable R^2 score for the model
# This value can be adjusted based on the expected performance
# of the Linear Regression model on the California Housing dataset.
R2_SCORE_THRESHOLD = 0.50

# A fixture to set up the trained model and test data once for all tests.
# The 'session' scope ensures the model is loaded only once per test run,
# which is more efficient.
@pytest.fixture(scope="session")
def trained_model_and_data():
    """
    Fixture to load the trained model and fetch test data for tests.
    """
    try:
        # Load the model saved by src/train.py
        model = retrieve_model("models/linear_regression_model.joblib")
        # Fetch and split the data
        _, X_test, _, y_test = fetch_data_split()
        return model, X_test, y_test
    except FileNotFoundError:
        # If the model is not found, skip the tests and provide an error message.
        pytest.fail("Model artifact 'models/linear_regression_model.joblib' not found. "
                    "Please run 'python src/train.py' first.")

def test_model_instance(trained_model_and_data):
    """
    Unit test to validate that the loaded artifact is a LinearRegression instance.
    """
    model, _, _ = trained_model_and_data
    assert isinstance(model, LinearRegression), "Loaded model is not a LinearRegression instance."

def test_model_is_trained(trained_model_and_data):
    """
    Unit test to check if the model has been trained by verifying
    the existence of coefficients and the intercept.
    """
    model, _, _ = trained_model_and_data
    assert hasattr(model, 'coef_'), "Model has not been trained: 'coef_' attribute is missing."
    assert hasattr(model, 'intercept_'), "Model has not been trained: 'intercept_' attribute is missing."

def test_dataset_loading():
    """
    Unit test to verify that the dataset is loaded and split correctly.
    This also implicitly tests the 'fetch_data_split' utility function.
    """
    _, X_test, _, y_test = fetch_data_split()
    assert X_test.shape[0] > 0, "Test dataset is empty."
    assert y_test.shape[0] > 0, "Test labels are empty."
    assert X_test.shape[0] == y_test.shape[0], "Mismatch in number of samples between features and labels."

def test_r2_score_threshold(trained_model_and_data):
    """
    Unit test to ensure the trained model's R^2 score on the test set
    exceeds a predefined minimum threshold.
    """
    model, X_test, y_test = trained_model_and_data
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"\n[TEST] R^2 Score on test set: {r2:.4f}")
    assert r2 > R2_SCORE_THRESHOLD, f"R^2 score ({r2:.4f}) is below the required threshold of {R2_SCORE_THRESHOLD}."

