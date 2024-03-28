# Use an official Miniconda runtime as a parent image
FROM continuumio/miniconda3:4.10.3

# Set the working directory in the container to /app
WORKDIR /app

# Install necessary system dependencies for building packages like psutil
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    python3-dev

# Copy the current directory contents into the container at /app
COPY . /app

# Create the environment using the environment.yml file
RUN conda env create -f /app/environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "m2-env", "/bin/bash", "-c"]

# Ensure Python output is set straight to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Define environment variable
ENV NAME m2-env

# Make sure the environment is activated
ENTRYPOINT ["conda", "run", "-n", "m2-env", "python"]

# Run main.py 
CMD ["/app/src/main.py"]