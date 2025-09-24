FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install only the dependencies needed for tests
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install \
    "numpy>=1.24.2" \
    "scipy>=1.11.3" \
    "Pillow>=9.4.0" \
    "faiss-cpu>=1.7.3" \
    "threatexchange>=1.0.13"

# Copy the project files
COPY . /app

# Install the package in editable mode without its heavy dependencies
RUN pip install --no-deps -e .

# Command to run the tests
CMD ["python", "-m", "unittest", "discover", "-s", "tests"]
