# src/quantize.py

import numpy as np
import joblib
import os
from utils import quantize_uint8, dequantize_uint8, retrieve_model, fetch_data_split, compute_scores

def quantize_main():
    """Entry point for quantization logic."""
    print("[QZ] Loading model artifact...")
    mdl = retrieve_model("models/linear_regression_model.joblib")

    # Extract params
    coeffs = mdl.coef_
    bias = mdl.intercept_

    print(f"[QZ] Coefficient shape: {coeffs.shape}")
    print(f"[QZ] Intercept: {bias}")
    print(f"[QZ] Coefficients (first 5): {coeffs[:5]}")

    # Save uncompressed params
    params_raw = {
        'coef': coeffs,
        'intercept': bias
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(params_raw, "models/unquant_params.joblib")
    print("\n[QZ] Raw parameters saved to models/unquant_params.joblib")

    # Quantize coefficients and bias to unsigned 8-bit
    print("\n[QZ] Quantizing parameters to unsigned 8-bit...")
    q_coef, scale_c, zero_point_c = quantize_uint8(coeffs)
    q_bias, scale_b, zero_point_b = quantize_uint8(np.array([bias]))
    
    # Save quantized params
    params_quant = {
        'quant_coef8': q_coef,
        'scale_coef8': scale_c,
        'zero_point_coef8': zero_point_c,
        'quant_intercept8': q_bias[0],
        'scale_intercept8': scale_b,
        'zero_point_intercept8': zero_point_b
    }
    joblib.dump(params_quant, "models/quant_params.joblib", compress=3)
    print("[QZ] Quantized params saved to models/quant_params.joblib")

    # Dequantize for local test
    d_coef = dequantize_uint8(q_coef, scale_c, zero_point_c)
    d_bias = dequantize_uint8(np.array([params_quant['quant_intercept8']]), params_quant['scale_intercept8'], params_quant['zero_point_intercept8'])[0]

    # Error checks
    err_coef = np.abs(coeffs - d_coef).max()
    err_bias = np.abs(bias - d_bias)
    print(f"\n[QZ] Max coef dequantization error: {err_coef:.8f}")
    print(f"[QZ] Bias dequantization error: {err_bias:.8f}")

    # Inference test
    _, X_te, _, y_te = fetch_data_split()
    y_pred_quant = X_te.values @ d_coef + d_bias

    # Print inference check
    print("\n[QZ] Inference check (first 5 predictions):")
    mdl_orig = retrieve_model("models/linear_regression_model.joblib")
    print(f"Original Model: {mdl_orig.predict(X_te[:5])}")
    print(f"Dequantized Model: {y_pred_quant[:5]}")
    
    diff = np.abs(mdl_orig.predict(X_te) - y_pred_quant)
    print(f"\n[QZ] Max prediction diff: {diff.max():.8f}")
    print(f"[QZ] Mean prediction diff: {diff.mean():.8f}")
    
    r2_quant, mse_quant = compute_scores(y_te, y_pred_quant)
    print(f"[QZ] R2 (quant): {r2_quant:.4f}")
    print(f"MSE (quant): {mse_quant:.4f}")

    if diff.max() < 0.1:
        print(f"[QZ] Quantization quality: good (max diff: {diff.max():.6f})")
    elif diff.max() < 1.0:
        print(f"[QZ] Quantization quality: ok (max diff: {diff.max():.6f})")
    else:
        print(f"[QZ] Quantization quality: poor (max diff: {diff.max():.6f})")

if __name__ == "__main__":
    quantize_main()
