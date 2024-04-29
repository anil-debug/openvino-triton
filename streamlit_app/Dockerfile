FROM python:3.8-slim

WORKDIR /app

# Copy requirements.txt from the parent directory
COPY requirements.txt .

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libb64-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app files into the container
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "main.py"]
