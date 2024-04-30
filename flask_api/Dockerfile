FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libb64-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt from the parent directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app files into the container
COPY . .

# Expose the Flask port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
