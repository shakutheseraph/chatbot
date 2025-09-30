# Use the official Python image as a base
FROM python:3.11-slim

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED True

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
# Ensure your FastAPI file is named gpt_back.py
COPY gpt_back.py /app/

# Cloud Run defaults to port 8080, so we use the PORT environment variable
ENV PORT 8080 

# Run Uvicorn with the application
# Format: uvicorn <filename>:<app_instance_name>
CMD ["uvicorn", "gpt_back:app", "--host", "0.0.0.0", "--port", "8080"]