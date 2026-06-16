# 1. Start from a base image that already has Python 3.11
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system packages needed for some Python libraries (like scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only requirements.txt first – this layer is cached if requirements don't change
COPY requirements.txt .
# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the project files
COPY . .

# 6. Command to run when the container starts
CMD ["python", "main.py"]