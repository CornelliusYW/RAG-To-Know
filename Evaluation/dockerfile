# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variables for MLflow and Neon PostgreSQL connection
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000
ENV POSTGRES_URI=YOUR_URI

# Run MLflow server when the container launches
CMD mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ${POSTGRES_URI} --default-artifact-root ./mlruns