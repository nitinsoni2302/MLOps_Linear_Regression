# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code to the container
COPY . .

# Define environment variable for Python unbuffered output (optional but good for logging)
ENV PYTHONUNBUFFERED 1

# Command to run your training script (you can change this to run other scripts as needed)
# This will be the default command when the container starts
CMD ["python", "src/train.py"]