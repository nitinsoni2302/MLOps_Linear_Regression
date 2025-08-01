# src/train.py

import os
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Create the models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def train_and_evaluate():
    """
    Fetches the dataset, trains a Linear Regression model,
    and saves the model and evaluation metrics.
    """
    print("Fetching California Housing dataset...")
    california_housing = fetch_california_housing(as_frame=True)
    X = california_housing.data
    y = california_housing.target
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples.")

    # Initialize and train the model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Save the trained model using the correct filename
    model_path = os.path.join('models', 'linear_regression_model.joblib')
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}")

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\n--- Model Performance ---")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
