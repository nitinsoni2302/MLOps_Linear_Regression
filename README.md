**MLOps Linear Regression Pipeline**
A complete MLOps pipeline for Linear Regression using California Housing dataset with training, testing, quantization, Dockerization, and CI/CD automation.

**Table of Contents**
1. Overview
2. Repository Setup
3. Project Structure
4. Installation & Setup
5. Usage
6. Model Performance
7. Testing
8. Docker Usage
9. CI/CD Pipeline
10. Quantization Details

**Overview**
This project implements a complete MLOps pipeline for Linear Regression using the California Housing dataset. The pipeline includes:

**Model Training: Linear Regression using scikit-learn**
1.Testing: Comprehensive unit tests with pytest
2.Quantization: Manual 8-bit quantization for model compression
3.Containerization: Docker support for deployment
4.CI/CD: Automated GitHub Actions workflow

**Repository Setup**
The repository was set up by first creating a local project folder, initializing a Git repository within it, and then linking it to an empty repository created on GitHub. Here are the exact steps followed:

**Step 1: Create a Local Project**
# Create a new project directory and navigate into it
mkdir MLOps_Linear_Regression
cd MLOps_Linear_Regression

# Create a virtual environment and activate it
python -m venv mlops_env
mlops_env\Scripts\Activate.ps1

# Create the project's directory structure and files
mkdir "src", "tests", ".github\workflows", "models" -Force
touch nul > "requirements.txt", "Dockerfile"

**Step 2: Initialize Git and Make First Commit**
# Initialize a new git repository in the current folder
git init

# Add all project files to the staging area
git add .

# Create the initial commit
git commit -m "Initial project structure and files"

**Step 3: Link to GitHub and Push**
# Create a new with name MLOps_Linear_Regression, empty repository on GitHub first and make it private initially.
# Then, link your local repository to the remote one.
git remote add origin https://github.com/nitinsoni2302/mlops-linear-regression.git

# Set the default branch to 'main' and push your first commit
git branch -M main
git push -u origin main

**Project Structure**
The repository is structured to modularize the different stages of the MLOps pipeline. The key components are:

mlops-linear-regression/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── train.py
│   ├── quantize.py
│   ├── predict.py
│   └── utils.py
├── tests/
│   └── test_train.py
├── Dockerfile
├── requirements.txt
└── README.md

**Installation & Setup**
Prerequisites
1.Python 3.x
2. Git
3. Docker

**To get started, install the required dependencies**
pip install -r requirements.txt

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

**Usage**
Training the Model
The src/train.py script trains the Linear Regression model on the California Housing dataset and saves the resulting artifact to the models/ directory.

# Run the training script
python src/train.py

**Expected Output:**

(mlops-env) c:\Users\nitin\Documents\MLOps_Linear_Regression>python src/train.py
Fetching California Housing dataset...
Training on 16512 samples, testing on 4128 samples.
Training Linear Regression model...
Model training complete.
Trained model saved to models\linear_regression_model.joblib

**--- Model Performance ---**
R^2 Score: 0.5758
Mean Squared Error (MSE): 0.5559

**Running Quantization**
The src/quantize.py script loads the trained model and performs a manual 8-bit quantization of its parameters.

# Run the quantization script
python src/quantize.py

**Expected Output:**

(mlops-env) c:\Users\nitin\Documents\MLOps_Linear_Regression>python src/quantize.py
[QZ] Loading model artifact...
[QZ] Coefficient shape: (8,)
[QZ] Intercept: -37.02327770606416
[QZ] Coefficients (first 5): [ 4.48674910e-01  9.72425752e-03 -1.23323343e-01  7.83144907e-01
 -2.02962058e-06]

[QZ] Raw parameters saved to models/unquant_params.joblib

[QZ] Quantizing parameters to unsigned 8-bit...
[QZ] Quantized params saved to models/quant_params.joblib

[QZ] Max coef dequantization error: 0.00124565
[QZ] Bias dequantization error: 0.14518932

[QZ] Inference check (first 5 predictions):
Original Model: [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725]
Dequantized Model: [0.91327412 1.95866328 2.90339757 3.0338709  2.79855629]

[QZ] Max prediction diff: 0.22542581
[QZ] Mean prediction diff: 0.19382916
[QZ] R2 (quant): 0.5489
MSE (quant): 0.5912
[QZ] Quantization quality: ok (max diff: 0.225426)

**Model Performance**
Performance Comparison Table
Metric

Original Model (Floating-Point)

Quantized Model (8-bit)

R² Score on Test Set

0.5758

0.5489

MSE

0.5559

0.5912

Max Prediction Difference

N/A

0.225426

Mean Prediction Difference

N/A

0.193829

Model Size

1
˜
 .5 KB (approx.)

0
˜
 .1 KB (approx.)

The results demonstrate a minor drop in the R 
2
  score due to the precision loss from 8-bit quantization. However, this is offset by a significant reduction in model size, which is highly beneficial for deployment on resource-constrained devices.

**Testing**
-Running Tests Locally: The test suite is written using pytest and validates the core components of the training pipeline.

# Run all tests
pytest

Expected Test Output:

(mlops-env) C:\Users\nitin\Documents\MLOps_Linear_Regression>pytest
======================================================= test session starts ========================================================
platform win32 -- Python 3.9.23, pytest-8.4.1, pluggy-1.6.0
rootdir: C:\Users\nitin\Documents\MLOps_Linear_Regression
collected 4 items

tests\test_train.py ....                                                    [100%]

======================================================== 4 passed in 5.10s =========================================================

**Docker Usage**
Building and Testing the Container
The Dockerfile creates a portable environment for the model and its dependencies. The CI/CD pipeline builds and tests this container automatically, but you can also do it locally.

# Build the Docker image
docker build -t nitinsoni2302/mlops-linear-regression .

# Run the container to execute the predict script for verification
docker run --rm nitinsoni2302/mlops-linear-regression python src/predict.py

**Container Output:**

(mlops-env) c:\Users\nitin\Documents\MLOps_Linear_Regression>python src/predict.py
Starting quantized model verification with src/predict.py...
Loading California Housing dataset...
Test data loaded: 4128 samples.
Quantized parameters loaded successfully from models\quant_params.joblib
Parameters de-quantized.
Predictions on test set completed.

**--- Sample Predictions (First 5) ---**
Index 0: Actual = 0.48, Predicted = 0.91
Index 1: Actual = 0.46, Predicted = 1.96
Index 2: Actual = 5.00, Predicted = 2.90
Index 3: Actual = 2.19, Predicted = 3.03
Index 4: Actual = 2.78, Predicted = 2.80

**--- Overall Performance Metrics (8-bit Quantized) ---**
R^2 Score: 0.5489
Mean Squared Error (MSE): 0.5912

Quantized model verification successful.


**CI/CD Pipeline**
The GitHub Actions workflow is defined in .github/workflows/ci.yml and automates the entire pipeline on every push to the main branch.

Pipeline Stages
train_and_test:
Installs system and Python dependencies.

**Runs src/train.py and src/quantize.py to create model artifacts.**

Executes the pytest test suite, which includes the tests defined in tests/test_train.py, to validate the model and its performance.

Uploads the model artifacts for the next stage.

build_and_test_container:

Depends on the successful completion of train_and_test.

Downloads the model artifacts.

Builds the Docker image.

Runs the container to verify the predict.py script executes successfully.

Pushes the final, validated image to Docker Hub.

**Git Commit and push**
- git add .
- git commit -m "feat: Complete MLOps pipeline with corrected quantization and CI/CD"
- git push origin main

**Workflow Status**
The status of the pipeline can be monitored directly in the Actions tab of your GitHub repository.


Quantization Details
Manual 8-bit Quantization Process
Our custom 8-bit quantization process manually converts the model's floating-point parameters into unsigned 8-bit integers (uint8) for a significant size reduction. The process involves:

Parameter Extraction: The coef_ and intercept_ values are extracted from the trained LinearRegression model.

Scaling and Mapping: A scale factor is calculated to map the range of floating-point values to the uint8 integer range [0, 255].

Storage: The quantized parameters and their scale factor are saved to a file (quant_params.joblib).

Dequantization: For inference, the stored uint8 values are loaded and converted back to floats using the saved scale factor to produce predictions.

The manual quantization formula for a given value is:

quantized_value = round((float_value / scale_factor) + offset)

The dequantization formula is:

dequantized_value = (quantized_value - offset) * scale_factor

This process demonstrates how to achieve model compression without relying on built-in libraries.# MLOps Pipeline for Linear Regression 
