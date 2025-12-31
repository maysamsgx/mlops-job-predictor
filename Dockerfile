# Use a lightweight python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any)
# RUN apt-get update && apt-get install -y gcc

# Install python dependencies first (Caching Layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY inference_service.py .
COPY train_pipeline.py . 
COPY voting_clf.joblib .

# Expose the service port
EXPOSE 8000

# Run the application
# We use the array form for ENTRYPOINT/CMD to ensure signals are passed correctly
CMD ["uvicorn", "inference_service:app", "--host", "0.0.0.0", "--port", "8000"]
