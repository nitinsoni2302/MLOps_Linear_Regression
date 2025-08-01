# src/utils.py

import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def retrieve_model(path: str):
    """Loads a model artifact from the specified path."""
    return joblib.load(path)

def fetch_data_split(test_size=0.2, random_state=42):
    """
    Fetches the California Housing dataset and splits it into
    training and test sets.
    """
    california_housing = fetch_california_housing(as_frame=True)
    X = california_housing.data
    y = california_housing.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_tr, X_te, y_tr, y_te

def compute_scores(y_true, y_pred):
    """Computes R^2 and Mean Squared Error."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse

def quantize_uint8(arr: np.ndarray):
    """
    Quantizes a float array to an unsigned 8-bit integer array.
    Returns the quantized array, and the scale/zero-point for de-quantization.
    """
    min_val, max_val = arr.min(), arr.max()
    
    # Handle the edge case where all values are the same
    if np.isclose(min_val, max_val):
        scale = max_val / 127.5 if max_val != 0 else 1.0
        zero_point = 127
    else:
        # Calculate scale and zero-point for unsigned 8-bit integers [0, 255]
        scale = (max_val - min_val) / 255.0
        zero_point = np.round(-min_val / scale).astype(np.int32)
    
    quantized_arr = np.round(arr / scale + zero_point).clip(0, 255).astype(np.uint8)
    
    return quantized_arr, scale, zero_point

def dequantize_uint8(arr: np.ndarray, scale: float, zero_point: float):
    """
    De-quantizes an unsigned 8-bit integer array back to a float array.
    """
    return (arr.astype(np.float32) - zero_point) * scale
