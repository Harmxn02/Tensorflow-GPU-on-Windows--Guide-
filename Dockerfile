# Use an official Python runtime as a parent image
FROM python:3.11.2-slim

# Set the working directory in the container
WORKDIR /workspace

# Install necessary dependencies including Git
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install jupyter jupyterlab numpy pandas matplotlib seaborn scikit-learn && \
    pip install tensorflow[and-cuda] polars[gpu]

# Expose Jupyter port
EXPOSE 8888

# Set the command to keep the container running for VS Code
CMD ["tail", "-f", "/dev/null"]
