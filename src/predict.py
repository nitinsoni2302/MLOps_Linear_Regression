# src/predict.py
import joblib
import os
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from utils import dequantize_uint8

# --- Paths for quantized parameters ---
QUANT_PARAMS_PATH = os.path.join('models', 'quant_params.joblib')

def predict_quantized_main():
    """
    Main function to load quantized parameters, de-quantize them,
    and perform predictions for verification.
    """
    print("Starting quantized model verification with src/predict.py...")

    # Load the dataset to get the test set
    print("Loading California Housing dataset...")
    california_housing = fetch_california_housing(as_frame=True)
    X = california_housing.data
    y = california_housing.target
    
    # Split data to get the same test set used during training
    # Use the same random_state as in src/train.py for consistency
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Test data loaded: {X_test.shape[0]} samples.")

    # Check if the quantized parameters file exists
    if not os.path.exists(QUANT_PARAMS_PATH):
        print(f"Error: Quantized parameters not found at {QUANT_PARAMS_PATH}. Please run src/quantize.py first.")
        return

    # Load the quantized parameters
    try:
        quant_params = joblib.load(QUANT_PARAMS_PATH)
        print(f"Quantized parameters loaded successfully from {QUANT_PARAMS_PATH}")
    except Exception as e:
        print(f"Error loading the quantized parameters: {e}")
        return
    
    # De-quantize the coefficients and bias
    try:
        d_coef = dequantize_uint8(quant_params['quant_coef8'], quant_params['scale_coef8'], quant_params['zero_point_coef8'])
        d_bias = dequantize_uint8(np.array([quant_params['quant_intercept8']]), quant_params['scale_intercept8'], quant_params['zero_point_intercept8'])[0]
        print("Parameters de-quantized.")
    except Exception as e:
        print(f"Error during de-quantization: {e}")
        return

    # Run predictions on the test set using the de-quantized parameters
    try:
        y_pred = X_test.values @ d_coef + d_bias
        print("Predictions on test set completed.")

        # Print sample outputs
        print("\n--- Sample Predictions (First 5) ---")
        for i in range(5):
            print(f"Index {i}: Actual = {y_test.iloc[i]:.2f}, Predicted = {y_pred[i]:.2f}")
        
        # Print overall performance metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print("\n--- Overall Performance Metrics (8-bit Quantized) ---")
        print(f"R^2 Score: {r2:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        
        print("\nQuantized model verification successful.")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    predict_quantized_main()
