# Use an official Python 3.12 runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install build essentials and clean up in one layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install setuptools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy just the requirements.txt file first
COPY requirements.txt .

# Install the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project
COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e .

# Run the serve.py script
CMD ["python", "-m", "pyannote_whisper.serve"]
